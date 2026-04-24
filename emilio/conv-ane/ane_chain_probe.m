// ane_chain_probe.m
//
// Question: can two compiled .mlmodelc artifacts be loaded into the ANE
// daemon, and can we then issue prepareChainingWithModel:options:chainingReq:qos:
// such that one model's output feeds into another's input — and importantly,
// can the chain be re-targeted per call (i.e., MoE-style "fire expert k_t for
// this token")?
//
// This probe doesn't try to actually execute a chain (we don't know the
// chainingReq schema). It does three concrete things:
//
//   1. Enumerate every selector on _ANEClient, _ANEDaemonConnection, _ANEModel,
//      _ANEProgramForEvaluation, _ANERequest containing "chain"/"prepare"/
//      "instance"/"newInstance" with full type-encoded signatures.
//
//   2. Build two _ANEModel objects from two existing compiled .mlmodelc paths
//      passed on argv. Compile + load each.
//
//   3. Resolve _ANEDaemonProtocol via NSXPCConnection metadata, print every
//      selector containing "chain"/"prepare"/"instance" along with the
//      NSMethodSignature recovered live from the protocol.
//
// Build:
//   clang -fobjc-arc -framework Foundation \
//     -o ane_chain_probe ane_chain_probe.m
//
// Run:
//   ./ane_chain_probe path/to/A.mlmodelc path/to/B.mlmodelc

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <objc/runtime.h>
#import <objc/message.h>

static NSString *const kAppleNeuralEnginePath =
    @"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine";

static void Log(NSString *fmt, ...) NS_FORMAT_FUNCTION(1, 2);
static void Log(NSString *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    NSString *s = [[NSString alloc] initWithFormat:fmt arguments:ap];
    va_end(ap);
    fprintf(stdout, "%s\n", s.UTF8String);
    fflush(stdout);
}

static BOOL LoadImage(NSString *path) {
    void *h = dlopen(path.fileSystemRepresentation, RTLD_NOW);
    if (!h) {
        Log(@"dlopen FAILED %@: %s", path, dlerror() ?: "unknown");
        return NO;
    }
    Log(@"dlopen OK    %@", path);
    return YES;
}

static NSArray<NSString *> *AllSelectors(Class cls, BOOL classMethods) {
    if (!cls) return @[];
    Class target = classMethods ? object_getClass((id)cls) : cls;
    unsigned int n = 0;
    Method *m = class_copyMethodList(target, &n);
    NSMutableArray *out = [NSMutableArray array];
    for (unsigned int i = 0; i < n; i++) {
        SEL s = method_getName(m[i]);
        if (s) [out addObject:NSStringFromSelector(s)];
    }
    free(m);
    return [out sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];
}

static BOOL MatchesAny(NSString *name, NSArray<NSString *> *needles) {
    NSString *low = name.lowercaseString;
    for (NSString *n in needles) if ([low containsString:n]) return YES;
    return NO;
}

static NSString *SigFor(id target, NSString *selName) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) return @"<unavailable>";
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig) return @"<no signature>";
    NSMutableArray *args = [NSMutableArray array];
    for (NSUInteger i = 2; i < sig.numberOfArguments; i++) {
        [args addObject:[NSString stringWithUTF8String:[sig getArgumentTypeAtIndex:i]]];
    }
    return [NSString stringWithFormat:@"return=%s args=[%@]",
            sig.methodReturnType, [args componentsJoinedByString:@", "]];
}

static void DumpClass(NSString *className, NSArray<NSString *> *needles) {
    Class cls = NSClassFromString(className);
    Log(@"\n=== %@ ===", className);
    if (!cls) { Log(@"  (class missing)"); return; }

    Log(@"  CLASS methods matching %@:", needles);
    for (NSString *m in AllSelectors(cls, YES)) {
        if (MatchesAny(m, needles)) {
            Log(@"    +[%@ %@]  %@", className, m, SigFor((id)cls, m));
        }
    }
    Log(@"  INSTANCE methods matching %@:", needles);
    // try to obtain a cheap instance for sig recovery
    id inst = nil;
    @try { inst = [[cls alloc] init]; } @catch (__unused id e) { inst = nil; }
    for (NSString *m in AllSelectors(cls, NO)) {
        if (MatchesAny(m, needles)) {
            Log(@"    -[%@ %@]  %@", className, m,
                inst ? SigFor(inst, m) : @"<no instance>");
        }
    }
}

static id Invoke(id target, NSString *selName, NSArray *args, NSString **err) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) {
        if (err) *err = @"<unavailable>";
        return nil;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != args.count + 2) {
        if (err) *err = @"<sig mismatch>";
        return nil;
    }
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = target;
    inv.selector = sel;
    for (NSUInteger i = 0; i < args.count; i++) {
        const char *t = [sig getArgumentTypeAtIndex:i + 2];
        if (t[0] != '@') { if (err) *err = [NSString stringWithFormat:@"arg %lu non-object %s", (unsigned long)i, t]; return nil; }
        __unsafe_unretained id a = args[i];
        [inv setArgument:&a atIndex:i + 2];
    }
    @try { [inv invoke]; }
    @catch (NSException *e) { if (err) *err = [NSString stringWithFormat:@"exception: %@", e.reason]; return nil; }
    if (sig.methodReturnType[0] == '@') {
        __unsafe_unretained id r = nil;
        [inv getReturnValue:&r];
        return r;
    }
    return nil;
}

// _ANEClient compileModel:options:qos:error:  / loadModel:options:qos:error:
static NSString *InvokeBoolErr(id client, NSString *selName, id model) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![client respondsToSelector:sel]) return @"<unavailable>";
    NSMethodSignature *sig = [client methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 6) return @"<sig mismatch>";
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = client; inv.selector = sel;
    __unsafe_unretained id m = model; [inv setArgument:&m atIndex:2];
    NSDictionary *opts = @{}; __unsafe_unretained id o = opts; [inv setArgument:&o atIndex:3];
    NSInteger qos = 0; [inv setArgument:&qos atIndex:4];
    NSError *__autoreleasing err = nil; NSError *__autoreleasing *errp = &err;
    [inv setArgument:&errp atIndex:5];
    @try { [inv invoke]; } @catch (NSException *e) { return [NSString stringWithFormat:@"exception: %@", e.reason]; }
    BOOL ret = NO; [inv getReturnValue:&ret];
    return [NSString stringWithFormat:@"return=%@ err=%@",
            ret ? @"YES" : @"NO",
            err ? [NSString stringWithFormat:@"code=%ld domain=%@ msg=%@", (long)err.code, err.domain, err.localizedDescription] : @"(nil)"];
}

static id BuildModel(NSURL *url) {
    Class modelClass = NSClassFromString(@"_ANEModel");
    NSString *err = nil;
    id m = Invoke((id)modelClass, @"modelAtURL:key:", @[ url, url.lastPathComponent ], &err);
    if (m) return m;
    Log(@"  modelAtURL:key: => %@", err ?: @"(nil)");
    err = nil;
    m = Invoke((id)modelClass, @"modelAtURL:key:modelAttributes:", @[ url, url.lastPathComponent, @{} ], &err);
    if (!m) Log(@"  modelAtURL:key:modelAttributes: => %@", err ?: @"(nil)");
    return m;
}

static void DumpDaemonProtocol(void) {
    Log(@"\n=== _ANEDaemonProtocol (live NSXPCInterface) ===");
    Class connClass = NSClassFromString(@"_ANEDaemonConnection");
    if (!connClass) { Log(@"  _ANEDaemonConnection missing"); return; }

    NSString *err = nil;
    id conn = Invoke((id)connClass, @"daemonConnection", @[], &err);
    if (!conn) { Log(@"  daemonConnection => %@", err ?: @"(nil)"); return; }
    id xpc = Invoke(conn, @"daemonConnection", @[], &err);
    if (!xpc) { Log(@"  -[_ANEDaemonConnection daemonConnection] => %@", err ?: @"(nil)"); return; }
    Log(@"  xpc connection: %@", [xpc description]);

    id iface = Invoke(xpc, @"remoteObjectInterface", @[], &err);
    if (!iface) { Log(@"  remoteObjectInterface => %@", err ?: @"(nil)"); return; }
    Log(@"  interface: %@", [iface description]);

    Protocol *proto = nil;
    SEL pSel = NSSelectorFromString(@"protocol");
    if ([iface respondsToSelector:pSel]) {
        proto = ((Protocol *(*)(id, SEL))objc_msgSend)(iface, pSel);
    }
    if (proto) {
        Log(@"  protocol name: %@", NSStringFromProtocol(proto));
        unsigned int n = 0;
        struct objc_method_description *list = protocol_copyMethodDescriptionList(proto, YES, YES, &n);
        Log(@"  required instance methods: %u", n);
        NSArray *needles = @[ @"chain", @"prepare", @"instance", @"new" ];
        for (unsigned int i = 0; i < n; i++) {
            NSString *nm = NSStringFromSelector(list[i].name);
            if (MatchesAny(nm, needles)) {
                Log(@"    -[%@ %@]  encoded_types=%s",
                    NSStringFromProtocol(proto), nm, list[i].types);
            }
        }
        free(list);
    } else {
        Log(@"  no protocol on interface");
    }
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s A.mlmodelc B.mlmodelc\n", argv[0]);
            return 2;
        }
        NSURL *urlA = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        NSURL *urlB = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]];

        Log(@"=== load private framework ===");
        if (!LoadImage(kAppleNeuralEnginePath)) return 1;

        // Always probe these for chain-related selectors
        NSArray *needles = @[ @"chain", @"prepare", @"instance", @"newinstance" ];
        for (NSString *cn in @[ @"_ANEClient", @"_ANEDaemonConnection",
                                @"_ANEModel", @"_ANEProgramForEvaluation",
                                @"_ANERequest", @"_ANEDeviceController",
                                @"_ANEInMemoryModel" ]) {
            DumpClass(cn, needles);
        }

        DumpDaemonProtocol();

        Log(@"\n=== build _ANEClient ===");
        Class clientClass = NSClassFromString(@"_ANEClient");
        NSString *err = nil;
        id client = Invoke((id)clientClass, @"sharedConnection", @[], &err);
        Log(@"client=%@", client ?: (err ?: @"(nil)"));
        if (!client) return 1;

        Log(@"\n=== build _ANEModel A: %@ ===", urlA.path);
        id mA = BuildModel(urlA);
        Log(@"  model A=%@", mA ?: @"(nil)");

        Log(@"\n=== build _ANEModel B: %@ ===", urlB.path);
        id mB = BuildModel(urlB);
        Log(@"  model B=%@", mB ?: @"(nil)");

        if (!mA || !mB) {
            Log(@"\nABORT: could not construct both _ANEModel objects.");
            return 1;
        }

        Log(@"\n=== compile + load A ===");
        Log(@"  compiledExistsBefore A: %@", InvokeBoolErr(client, @"compiledModelExistsFor:", mA));
        Log(@"  compileModel: A => %@", InvokeBoolErr(client, @"compileModel:options:qos:error:", mA));
        Log(@"  loadModel:    A => %@", InvokeBoolErr(client, @"loadModel:options:qos:error:", mA));

        Log(@"\n=== compile + load B ===");
        Log(@"  compiledExistsBefore B: %@", InvokeBoolErr(client, @"compiledModelExistsFor:", mB));
        Log(@"  compileModel: B => %@", InvokeBoolErr(client, @"compileModel:options:qos:error:", mB));
        Log(@"  loadModel:    B => %@", InvokeBoolErr(client, @"loadModel:options:qos:error:", mB));

        // Inspect what _ANEClient now reports about its connections & loaded models.
        Log(@"\n=== _ANEClient state after load ===");
        for (NSString *sel in @[ @"connections", @"connectionsUsedForLoadingModels",
                                 @"isAnetoolRootDaemonConnection", @"isRootDaemon" ]) {
            err = nil;
            id v = Invoke(client, sel, @[], &err);
            Log(@"  -[_ANEClient %@] => %@", sel, v ?: (err ?: @"(nil)"));
        }
    }
    return 0;
}
