// ane_class_dump.m
//
// Live-runtime ObjC reflection of the AppleNeuralEngine private framework.
// Walks every class registered in the loaded image and prints, for any
// class whose name matches a needle list:
//   - all ivars with their @encode types and offsets
//   - all instance methods (selector + return type + arg types)
//   - all class methods
//   - all adopted protocols
//
// Goal: discover the class behind the `chainingReq` argument of
//   prepareChainingWithModel:options:chainingReq:qos:withReply:
// so we know how to synthesize one.
//
// Build:
//   clang -fobjc-arc -framework Foundation -o ane_class_dump ane_class_dump.m
// Run:
//   ./ane_class_dump

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <objc/runtime.h>

static NSString *const kAppleNeuralEnginePath =
    @"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine";
static NSString *const kANECompilerPath =
    @"/System/Library/PrivateFrameworks/ANECompiler.framework/Versions/A/ANECompiler";

static void Log(NSString *fmt, ...) NS_FORMAT_FUNCTION(1, 2);
static void Log(NSString *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    NSString *s = [[NSString alloc] initWithFormat:fmt arguments:ap];
    va_end(ap);
    fprintf(stdout, "%s\n", s.UTF8String); fflush(stdout);
}

static BOOL LoadImage(NSString *p) {
    if (!dlopen(p.fileSystemRepresentation, RTLD_NOW)) {
        Log(@"dlopen FAILED %@: %s", p, dlerror() ?: "?"); return NO;
    }
    Log(@"dlopen OK %@", p); return YES;
}

static BOOL Matches(NSString *name, NSArray<NSString *> *needles) {
    NSString *low = name.lowercaseString;
    for (NSString *n in needles) if ([low containsString:n]) return YES;
    return NO;
}

static NSString *EncodeStr(const char *t) { return t ? [NSString stringWithUTF8String:t] : @"?"; }

static NSString *MethodArgs(Method m) {
    unsigned int n = method_getNumberOfArguments(m);
    NSMutableArray *a = [NSMutableArray array];
    for (unsigned int i = 2; i < n; i++) {
        char buf[256]; buf[0] = '\0';
        method_getArgumentType(m, i, buf, sizeof(buf));
        [a addObject:EncodeStr(buf[0] ? buf : "?")];
    }
    return [a componentsJoinedByString:@", "];
}

static void DumpClass(Class cls) {
    const char *name = class_getName(cls);
    Class super = class_getSuperclass(cls);
    Log(@"\n#### %s   (super=%s,  size=%zu)", name,
        super ? class_getName(super) : "(nil)",
        class_getInstanceSize(cls));

    // adopted protocols
    unsigned int pn = 0;
    Protocol * __unsafe_unretained *protos = class_copyProtocolList(cls, &pn);
    if (pn) {
        NSMutableArray *ps = [NSMutableArray array];
        for (unsigned int i = 0; i < pn; i++) [ps addObject:NSStringFromProtocol(protos[i])];
        Log(@"  protocols: %@", [ps componentsJoinedByString:@", "]);
    }
    free(protos);

    // ivars
    unsigned int in = 0;
    Ivar *iv = class_copyIvarList(cls, &in);
    if (in) {
        Log(@"  ivars (%u):", in);
        for (unsigned int i = 0; i < in; i++) {
            const char *ivname = ivar_getName(iv[i]);
            const char *ivtype = ivar_getTypeEncoding(iv[i]);
            ptrdiff_t off = ivar_getOffset(iv[i]);
            Log(@"    [+0x%04zx] %s   %s",
                (size_t)off, ivname ?: "(?)", ivtype ?: "(?)");
        }
    }
    free(iv);

    // instance methods
    unsigned int mn = 0;
    Method *mm = class_copyMethodList(cls, &mn);
    if (mn) {
        Log(@"  instance methods (%u):", mn);
        // Sort selectors lexicographically
        NSMutableArray *items = [NSMutableArray arrayWithCapacity:mn];
        for (unsigned int i = 0; i < mn; i++) {
            char ret[256]; method_getReturnType(mm[i], ret, sizeof(ret));
            [items addObject:[NSString stringWithFormat:@"    -[%@]  ret=%@  args=[%@]",
                              NSStringFromSelector(method_getName(mm[i])),
                              EncodeStr(ret),
                              MethodArgs(mm[i])]];
        }
        [items sortUsingSelector:@selector(localizedStandardCompare:)];
        for (NSString *s in items) Log(@"%@", s);
    }
    free(mm);

    // class methods (methods on metaclass)
    Class meta = object_getClass((id)cls);
    unsigned int cn = 0;
    Method *cm = class_copyMethodList(meta, &cn);
    if (cn) {
        Log(@"  class methods (%u):", cn);
        NSMutableArray *items = [NSMutableArray arrayWithCapacity:cn];
        for (unsigned int i = 0; i < cn; i++) {
            char ret[256]; method_getReturnType(cm[i], ret, sizeof(ret));
            [items addObject:[NSString stringWithFormat:@"    +[%@]  ret=%@  args=[%@]",
                              NSStringFromSelector(method_getName(cm[i])),
                              EncodeStr(ret),
                              MethodArgs(cm[i])]];
        }
        [items sortUsingSelector:@selector(localizedStandardCompare:)];
        for (NSString *s in items) Log(@"%@", s);
    }
    free(cm);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        LoadImage(kAppleNeuralEnginePath);
        LoadImage(kANECompilerPath);

        NSArray<NSString *> *needles = @[ @"chain", @"chaining", @"request",
                                          @"instance", @"inmemory", @"program" ];
        if (argc > 1) {
            NSMutableArray *m = [NSMutableArray array];
            for (int i = 1; i < argc; i++) [m addObject:[[NSString stringWithUTF8String:argv[i]] lowercaseString]];
            needles = m;
        }
        Log(@"=== needles: %@", needles);

        unsigned int n = 0;
        Class *all = (Class *)objc_copyClassList(&n);
        Log(@"=== %u total ObjC classes registered", n);

        NSMutableArray<NSString *> *matched = [NSMutableArray array];
        for (unsigned int i = 0; i < n; i++) {
            NSString *cn = NSStringFromClass(all[i]);
            if (Matches(cn, needles)) [matched addObject:cn];
        }
        [matched sortUsingSelector:@selector(localizedStandardCompare:)];
        Log(@"=== %lu matching classes:", (unsigned long)matched.count);
        for (NSString *s in matched) Log(@"  - %@", s);

        for (NSString *cn in matched) {
            Class c = NSClassFromString(cn);
            if (c) DumpClass(c);
        }
        free(all);
    }
    return 0;
}
