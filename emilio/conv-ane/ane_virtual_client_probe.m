// ane_virtual_client_probe.m
//
// Minimal private-runtime admissibility probe for the direct ANE IOKit path.
// It does not compile or load a model. It only answers whether an unsigned dev
// binary can obtain an _ANEVirtualClient connection and query basic negotiated
// capability fields.
//
// Build:
//   clang -fobjc-arc -framework Foundation \
//     -o /tmp/ane_virtual_client_probe emilio/conv-ane/ane_virtual_client_probe.m
//
// Run:
//   /tmp/ane_virtual_client_probe
//   /tmp/ane_virtual_client_probe --summary tmp/ane_virtual_client_probe/summary.json

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <objc/runtime.h>

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
    void *handle = dlopen(path.fileSystemRepresentation, RTLD_NOW);
    if (!handle) {
        Log(@"dlopen FAILED %@: %s", path, dlerror() ?: "unknown");
        return NO;
    }
    Log(@"dlopen OK    %@", path);
    return YES;
}

static NSArray<NSString *> *AllSelectors(Class cls, BOOL classMethods) {
    if (!cls) return @[];
    Class target = classMethods ? object_getClass((id)cls) : cls;
    unsigned int count = 0;
    Method *methods = class_copyMethodList(target, &count);
    NSMutableArray<NSString *> *names = [NSMutableArray arrayWithCapacity:count];
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        if (sel) [names addObject:NSStringFromSelector(sel)];
    }
    free(methods);
    return [names sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];
}

static BOOL ContainsAny(NSString *name, NSArray<NSString *> *needles) {
    NSString *low = name.lowercaseString;
    for (NSString *needle in needles) {
        if ([low containsString:needle]) return YES;
    }
    return NO;
}

static NSString *SignatureString(id target, NSString *selName) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) return @"<unavailable>";
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig) return @"<no signature>";
    NSMutableArray<NSString *> *args = [NSMutableArray array];
    for (NSUInteger i = 2; i < sig.numberOfArguments; i++) {
        [args addObject:[NSString stringWithUTF8String:[sig getArgumentTypeAtIndex:i]]];
    }
    return [NSString stringWithFormat:@"return=%s args=[%@]",
            sig.methodReturnType, [args componentsJoinedByString:@", "]];
}

static id InvokeObjectNoArg(id target, NSString *selName, NSString **error) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) {
        if (error) *error = @"<unavailable>";
        return nil;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 2 || sig.methodReturnType[0] != '@') {
        if (error) *error = @"<unsupported signature>";
        return nil;
    }
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = target;
    inv.selector = sel;
    @try {
        [inv invoke];
    } @catch (NSException *exception) {
        if (error) *error = [NSString stringWithFormat:@"exception: %@", exception.reason];
        return nil;
    }
    __unsafe_unretained id result = nil;
    [inv getReturnValue:&result];
    return result;
}

static id InvokeObjectUInt32(id target, NSString *selName, uint32_t value, NSString **error) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) {
        if (error) *error = @"<unavailable>";
        return nil;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 3 || sig.methodReturnType[0] != '@') {
        if (error) *error = @"<unsupported signature>";
        return nil;
    }
    const char *argType = [sig getArgumentTypeAtIndex:2];
    if (!(argType[0] == 'I' || argType[0] == 'i' || argType[0] == 'S' || argType[0] == 's' ||
          argType[0] == 'L' || argType[0] == 'l' || argType[0] == 'Q' || argType[0] == 'q' ||
          argType[0] == 'C' || argType[0] == 'c')) {
        if (error) *error = [NSString stringWithFormat:@"<unsupported arg type %s>", argType];
        return nil;
    }
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = target;
    inv.selector = sel;
    unsigned long long boxed = value;
    [inv setArgument:&boxed atIndex:2];
    @try {
        [inv invoke];
    } @catch (NSException *exception) {
        if (error) *error = [NSString stringWithFormat:@"exception: %@", exception.reason];
        return nil;
    }
    __unsafe_unretained id result = nil;
    [inv getReturnValue:&result];
    return result;
}

static id InvokeNoArgValue(id target, NSString *selName, NSString **error) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) {
        if (error) *error = @"<unavailable>";
        return nil;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 2) {
        if (error) *error = @"<unsupported signature>";
        return nil;
    }
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = target;
    inv.selector = sel;
    @try {
        [inv invoke];
    } @catch (NSException *exception) {
        if (error) *error = [NSString stringWithFormat:@"exception: %@", exception.reason];
        return nil;
    }

    const char *ret = sig.methodReturnType;
    if (ret[0] == 'v') return @"<void>";
    if (ret[0] == '@') {
        __unsafe_unretained id obj = nil;
        [inv getReturnValue:&obj];
        return obj ?: [NSNull null];
    }
    if (ret[0] == 'B' || ret[0] == 'c' || ret[0] == 'C') {
        unsigned char value = 0;
        [inv getReturnValue:&value];
        return @(value != 0);
    }
    if (ret[0] == 'i' || ret[0] == 's' || ret[0] == 'l' || ret[0] == 'q') {
        long long value = 0;
        [inv getReturnValue:&value];
        return @(value);
    }
    if (ret[0] == 'I' || ret[0] == 'S' || ret[0] == 'L' || ret[0] == 'Q') {
        unsigned long long value = 0;
        [inv getReturnValue:&value];
        return @(value);
    }
    if (error) *error = [NSString stringWithFormat:@"<unsupported return type %s>", ret];
    return nil;
}

static NSDictionary *CallMethod(id target, NSString *selName) {
    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    out[@"selector"] = selName;
    out[@"signature"] = SignatureString(target, selName);

    NSString *error = nil;
    id value = InvokeNoArgValue(target, selName, &error);
    if (error) out[@"error"] = error;
    if (value) {
        out[@"value"] = value;
        if ([value isKindOfClass:[NSObject class]] && ![value isKindOfClass:[NSNumber class]] && ![value isKindOfClass:[NSString class]] && ![value isKindOfClass:[NSNull class]]) {
            out[@"value_description"] = [value description];
        }
    }
    return out;
}

static NSDictionary *TryCreateClient(Class cls) {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    NSMutableArray *candidates = [NSMutableArray array];
    NSMutableArray *attempts = [NSMutableArray array];

    NSArray<NSString *> *classMethods = AllSelectors(cls, YES);
    NSArray<NSString *> *needles = @[ @"shared", @"connection", @"qos", @"qos:" ];
    for (NSString *name in classMethods) {
        if (!ContainsAny(name, needles)) continue;
        [candidates addObject:@{ @"selector": name, @"signature": SignatureString((id)cls, name) }];
    }
    result[@"candidate_class_methods"] = candidates;

    id client = nil;
    for (NSString *name in classMethods) {
        NSString *low = name.lowercaseString;
        if (![low containsString:@"shared"] || ![low containsString:@"connection"]) continue;
        NSMethodSignature *sig = [(id)cls methodSignatureForSelector:NSSelectorFromString(name)];
        if (!sig || sig.methodReturnType[0] != '@') continue;

        NSMutableDictionary *attempt = [NSMutableDictionary dictionary];
        attempt[@"selector"] = name;
        attempt[@"signature"] = SignatureString((id)cls, name);

        NSString *error = nil;
        id value = nil;
        if (sig.numberOfArguments == 2) {
            value = InvokeObjectNoArg((id)cls, name, &error);
        } else if (sig.numberOfArguments == 3) {
            value = InvokeObjectUInt32((id)cls, name, 0, &error);
            attempt[@"arg0"] = @0;
        } else {
            attempt[@"error"] = @"<skipped unsupported arity>";
            [attempts addObject:attempt];
            continue;
        }

        if (error) attempt[@"error"] = error;
        if (value) {
            attempt[@"result_class"] = NSStringFromClass([value class]);
            attempt[@"result_description"] = [value description];
            client = value;
            [attempts addObject:attempt];
            break;
        }
        [attempts addObject:attempt];
    }

    result[@"constructor_attempts"] = attempts;
    if (client) {
        result[@"connected"] = @YES;
        result[@"client_class"] = NSStringFromClass([client class]);
        result[@"client_description"] = [client description];
        result[@"client"] = client;
    } else {
        result[@"connected"] = @NO;
    }
    return result;
}

static NSString *SummaryPathFromArgs(int argc, const char *argv[]) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--summary") == 0) {
            return [NSString stringWithUTF8String:argv[i + 1]];
        }
    }
    return nil;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSString *summaryPath = SummaryPathFromArgs(argc, argv);

        NSMutableDictionary *summary = [NSMutableDictionary dictionary];
        summary[@"probe"] = @"ane_virtual_client_probe";
        summary[@"framework_path"] = kAppleNeuralEnginePath;

        Log(@"=== load private framework ===");
        BOOL loaded = LoadImage(kAppleNeuralEnginePath);
        summary[@"framework_loaded"] = @(loaded);
        if (!loaded) {
            if (summaryPath) {
                NSData *json = [NSJSONSerialization dataWithJSONObject:summary options:NSJSONWritingPrettyPrinted error:nil];
                [json writeToFile:summaryPath atomically:YES];
            }
            return 1;
        }

        Log(@"\n=== resolve _ANEVirtualClient ===");
        Class cls = NSClassFromString(@"_ANEVirtualClient");
        summary[@"class_present"] = @(cls != Nil);
        if (!cls) {
            Log(@"_ANEVirtualClient missing");
            if (summaryPath) {
                NSData *json = [NSJSONSerialization dataWithJSONObject:summary options:NSJSONWritingPrettyPrinted error:nil];
                [json writeToFile:summaryPath atomically:YES];
            }
            return 2;
        }

        NSDictionary *connect = TryCreateClient(cls);
        NSMutableDictionary *connectSummary = [connect mutableCopy];
        id client = connectSummary[@"client"];
        [connectSummary removeObjectForKey:@"client"];
        summary[@"connection"] = connectSummary;

        Log(@"class present: YES");
        Log(@"connected: %@", [connectSummary[@"connected"] boolValue] ? @"YES" : @"NO");
        for (NSDictionary *attempt in connectSummary[@"constructor_attempts"]) {
            Log(@"  %@ -> %@%@",
                attempt[@"selector"],
                attempt[@"result_class"] ?: @"nil",
                attempt[@"error"] ? [@" error=" stringByAppendingString:attempt[@"error"]] : @"");
        }

        if (client) {
            Log(@"\n=== negotiated state ===");
            NSArray<NSString *> *calls = @[
                @"exchangeBuildVersionInfo",
                @"hostBuildVersionStr",
                @"validateEnvironmentForPrecompiledBinarySupport",
                @"negotiatedCapabilityMask",
                @"negotiatedDataInterfaceVersion",
                @"beginRealTimeTask",
                @"endRealTimeTask",
            ];
            NSMutableArray *callResults = [NSMutableArray array];
            for (NSString *name in calls) {
                NSDictionary *call = CallMethod(client, name);
                [callResults addObject:call];
                Log(@"  %@ => %@%@",
                    name,
                    call[@"value_description"] ?: call[@"value"] ?: @"(nil)",
                    call[@"error"] ? [@" error=" stringByAppendingString:call[@"error"]] : @"");
            }
            summary[@"instance_calls"] = callResults;
        }

        if (summaryPath) {
            NSString *dir = [summaryPath stringByDeletingLastPathComponent];
            if (dir.length > 0) {
                [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
            }
            NSError *jsonError = nil;
            NSData *json = [NSJSONSerialization dataWithJSONObject:summary options:NSJSONWritingPrettyPrinted error:&jsonError];
            if (!json) {
                Log(@"failed to encode summary JSON: %@", jsonError.localizedDescription);
                return 3;
            }
            if (![json writeToFile:summaryPath atomically:YES]) {
                Log(@"failed to write summary: %@", summaryPath);
                return 4;
            }
            Log(@"\nsummary: %@", summaryPath);
        }
    }

    return 0;
}