// ane_inmemory_model_probe.m
//
// Minimal daemon-backed private-runtime probe for _ANEInMemoryModelDescriptor.
// It answers two bounded questions:
//   1. Can a trivial inline-constant MIL program compile and load in memory?
//   2. For a one-weight BLOBFILE MIL program, which descriptor weights value
//      shape gets farther: raw NSData or _ANEWeight?
//
// Build:
//   clang -fobjc-arc -framework Foundation \
//     -o /tmp/ane_inmemory_model_probe emilio/conv-ane/ane_inmemory_model_probe.m
//
// Run:
//   /tmp/ane_inmemory_model_probe
//   /tmp/ane_inmemory_model_probe --summary tmp/ane_inmemory_model_probe/summary.json

#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#import <dlfcn.h>
#import <objc/runtime.h>

static NSString *const kAppleNeuralEnginePath =
    @"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine";

static NSString *const kReplayMILPath =
    @"/Users/alvarovidela/Code/em2/python/moe/out/qwen36/conv_probe_20260423_B1_E000/qwen36_L00_expert000_B1_conv_int4.mlmodelc/model.mil";

static NSString *const kReplayWeightPath =
    @"/Users/alvarovidela/Code/em2/python/moe/out/qwen36/conv_probe_20260423_B1_E000/qwen36_L00_expert000_B1_conv_int4.mlmodelc/weights/weight.bin";

static NSString *const kReplayCoreMLDataPath =
    @"/Users/alvarovidela/Code/em2/python/moe/out/qwen36/conv_probe_20260423_B1_E000/qwen36_L00_expert000_B1_conv_int4.mlmodelc/coremldata.bin";

static NSString *const kReplayAnalyticsCoreMLDataPath =
    @"/Users/alvarovidela/Code/em2/python/moe/out/qwen36/conv_probe_20260423_B1_E000/qwen36_L00_expert000_B1_conv_int4.mlmodelc/analytics/coremldata.bin";

static NSString *const kProbeSetCompilerOptionsFileNameKey =
    @"__copilot_setCompilerOptionsFileName";

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

static NSString *SummaryPathFromArgs(int argc, const char *argv[]) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--summary") == 0) {
            return [NSString stringWithUTF8String:argv[i + 1]];
        }
    }
    return nil;
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

static NSData *EmptyOptionsPlist(void) {
    NSError *error = nil;
    NSData *data = [NSPropertyListSerialization dataWithPropertyList:@{}
                                                              format:NSPropertyListBinaryFormat_v1_0
                                                             options:0
                                                               error:&error];
    if (!data) {
        Log(@"failed to create options plist: %@", error.localizedDescription);
    }
    return data;
}

static NSData *InlineMILText(void) {
    NSString *text =
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3520.4.1\"}, {\"coremlc-version\", \"3520.5.1\"}})]\n"
        @"{\n"
        @"    func main<ios18>(tensor<fp16, [1, 4]> x) {\n"
        @"            fp16 one = const()[name = string(\"one\"), val = fp16(0x1p+0)];\n"
        @"            tensor<fp16, [1, 4]> y = mul(x = x, y = one)[name = string(\"y\")];\n"
        @"        } -> (y);\n"
        @"}\n";
    return [text dataUsingEncoding:NSUTF8StringEncoding];
}

static NSData *BlobMILText(void) {
    NSString *text =
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3520.4.1\"}, {\"coremlc-version\", \"3520.5.1\"}})]\n"
        @"{\n"
        @"    func main<ios18>(tensor<fp16, [1, 4]> x) {\n"
        @"            tensor<fp16, [1, 4]> w = const()[name = string(\"w\"), val = tensor<fp16, [1, 4]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(0)))];\n"
        @"            tensor<fp16, [1, 4]> y = mul(x = x, y = w)[name = string(\"y\")];\n"
        @"        } -> (y);\n"
        @"}\n";
    return [text dataUsingEncoding:NSUTF8StringEncoding];
}

static NSData *RewriteMILWeightPath(NSData *milText, NSString *fromPath, NSString *toPath) {
    NSString *text = [[NSString alloc] initWithData:milText encoding:NSUTF8StringEncoding];
    if (!text) return nil;
    NSString *rewritten = [text stringByReplacingOccurrencesOfString:fromPath withString:toPath];
    return [rewritten dataUsingEncoding:NSUTF8StringEncoding];
}

static NSData *WeightBlobData(void) {
    const uint16_t halfs[4] = { 0x3c00, 0x4000, 0x4200, 0x4400 };
    return [NSData dataWithBytes:halfs length:sizeof(halfs)];
}

static NSData *SHA256(NSData *data) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    return [NSData dataWithBytes:digest length:sizeof(digest)];
}

static NSData *ReadFileData(NSString *path, NSMutableDictionary *caseSummary, NSString *fieldPrefix) {
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:path options:0 error:&error];
    NSString *pathKey = [NSString stringWithFormat:@"%@_path", fieldPrefix];
    NSString *bytesKey = [NSString stringWithFormat:@"%@_bytes", fieldPrefix];
    NSString *errorKey = [NSString stringWithFormat:@"%@_error", fieldPrefix];
    caseSummary[pathKey] = path;
    if (!data) {
        caseSummary[errorKey] = error.localizedDescription ?: @"read failed";
        return nil;
    }
    caseSummary[bytesKey] = @(data.length);
    return data;
}

static id InvokeDescriptorFactory(Class descriptorClass,
                                  NSString *selectorName,
                                  NSData *modelData,
                                  NSDictionary *weights,
                                  NSData *optionsPlist,
                                  NSString **error) {
    SEL sel = NSSelectorFromString(selectorName);
    if (!sel || ![(id)descriptorClass respondsToSelector:sel]) {
        if (error) *error = @"<unavailable>";
        return nil;
    }
    NSMethodSignature *sig = [(id)descriptorClass methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 5 || sig.methodReturnType[0] != '@') {
        if (error) *error = @"<unsupported signature>";
        return nil;
    }

    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = (id)descriptorClass;
    inv.selector = sel;
    __unsafe_unretained id arg0 = modelData;
    __unsafe_unretained id arg1 = weights;
    __unsafe_unretained id arg2 = optionsPlist;
    [inv setArgument:&arg0 atIndex:2];
    [inv setArgument:&arg1 atIndex:3];
    [inv setArgument:&arg2 atIndex:4];
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

static id InitInMemoryModel(Class modelClass, id descriptor, NSString **error) {
    id instance = [modelClass alloc];
    SEL sel = NSSelectorFromString(@"initWithDesctiptor:");
    if (!sel || ![instance respondsToSelector:sel]) {
        if (error) *error = @"<unavailable>";
        return nil;
    }
    NSMethodSignature *sig = [instance methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 3 || sig.methodReturnType[0] != '@') {
        if (error) *error = @"<unsupported signature>";
        return nil;
    }
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = instance;
    inv.selector = sel;
    __unsafe_unretained id arg = descriptor;
    [inv setArgument:&arg atIndex:2];
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

static BOOL InvokeBoolQoS(id target,
                          NSString *selName,
                          uint32_t qos,
                          NSDictionary *options,
                          NSError **outError,
                          NSString **errorText) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) {
        if (errorText) *errorText = @"<unavailable>";
        return NO;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 5) {
        if (errorText) *errorText = @"<unsupported signature>";
        return NO;
    }

    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = target;
    inv.selector = sel;
    unsigned int qosValue = qos;
    __unsafe_unretained id opts = options ?: @{};
    NSError *__autoreleasing error = nil;
    NSError *__autoreleasing *errorPtr = &error;
    [inv setArgument:&qosValue atIndex:2];
    [inv setArgument:&opts atIndex:3];
    [inv setArgument:&errorPtr atIndex:4];
    @try {
        [inv invoke];
    } @catch (NSException *exception) {
        if (errorText) *errorText = [NSString stringWithFormat:@"exception: %@", exception.reason];
        return NO;
    }

    BOOL ok = NO;
    if (sig.methodReturnType[0] == 'B' || sig.methodReturnType[0] == 'c' || sig.methodReturnType[0] == 'C') {
        [inv getReturnValue:&ok];
    } else {
        if (errorText) *errorText = [NSString stringWithFormat:@"<unsupported return type %s>", sig.methodReturnType];
        return NO;
    }
    if (outError) *outError = error;
    return ok;
}

static id InvokeObjectNoArg(id target, NSString *selName, NSString **errorText) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) {
        if (errorText) *errorText = @"<unavailable>";
        return nil;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 2 || sig.methodReturnType[0] != '@') {
        if (errorText) *errorText = @"<unsupported signature>";
        return nil;
    }
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = target;
    inv.selector = sel;
    @try {
        [inv invoke];
    } @catch (NSException *exception) {
        if (errorText) *errorText = [NSString stringWithFormat:@"exception: %@", exception.reason];
        return nil;
    }
    __unsafe_unretained id result = nil;
    [inv getReturnValue:&result];
    return result;
}

static id InvokeObjectWithObjectBool(id target,
                                     NSString *selName,
                                     id objectArg,
                                     BOOL boolArg,
                                     NSString **errorText) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) {
        if (errorText) *errorText = @"<unavailable>";
        return nil;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 4 || sig.methodReturnType[0] != '@') {
        if (errorText) *errorText = @"<unsupported signature>";
        return nil;
    }
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = target;
    inv.selector = sel;
    __unsafe_unretained id arg0 = objectArg;
    BOOL arg1 = boolArg;
    [inv setArgument:&arg0 atIndex:2];
    [inv setArgument:&arg1 atIndex:3];
    @try {
        [inv invoke];
    } @catch (NSException *exception) {
        if (errorText) *errorText = [NSString stringWithFormat:@"exception: %@", exception.reason];
        return nil;
    }
    __unsafe_unretained id result = nil;
    [inv getReturnValue:&result];
    return result;
}

static id InvokeObjectWithTwoObjects(id target,
                                     NSString *selName,
                                     id objectArg0,
                                     id objectArg1,
                                     NSString **errorText) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) {
        if (errorText) *errorText = @"<unavailable>";
        return nil;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 4 || sig.methodReturnType[0] != '@') {
        if (errorText) *errorText = @"<unsupported signature>";
        return nil;
    }
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = target;
    inv.selector = sel;
    __unsafe_unretained id arg0 = objectArg0;
    __unsafe_unretained id arg1 = objectArg1;
    [inv setArgument:&arg0 atIndex:2];
    [inv setArgument:&arg1 atIndex:3];
    @try {
        [inv invoke];
    } @catch (NSException *exception) {
        if (errorText) *errorText = [NSString stringWithFormat:@"exception: %@", exception.reason];
        return nil;
    }
    __unsafe_unretained id result = nil;
    [inv getReturnValue:&result];
    return result;
}

static BOOL InvokeVoidWithObject(id target, NSString *selName, id objectArg, NSString **errorText) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) {
        if (errorText) *errorText = @"<unavailable>";
        return NO;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 3 || sig.methodReturnType[0] != 'v') {
        if (errorText) *errorText = @"<unsupported signature>";
        return NO;
    }
    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = target;
    inv.selector = sel;
    __unsafe_unretained id arg = objectArg;
    [inv setArgument:&arg atIndex:2];
    @try {
        [inv invoke];
    } @catch (NSException *exception) {
        if (errorText) *errorText = [NSString stringWithFormat:@"exception: %@", exception.reason];
        return NO;
    }
    return YES;
}

static BOOL InvokeBoolWithObjectObjectQoS(id target,
                                          NSString *selName,
                                          id objectArg,
                                          NSDictionary *options,
                                          uint32_t qos,
                                          NSError **outError,
                                          NSString **errorText) {
    SEL sel = NSSelectorFromString(selName);
    if (!sel || ![target respondsToSelector:sel]) {
        if (errorText) *errorText = @"<unavailable>";
        return NO;
    }
    NSMethodSignature *sig = [target methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 6) {
        if (errorText) *errorText = @"<unsupported signature>";
        return NO;
    }

    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = target;
    inv.selector = sel;
    __unsafe_unretained id modelArg = objectArg;
    __unsafe_unretained id optionsArg = options ?: @{};
    unsigned int qosValue = qos;
    NSError *__autoreleasing error = nil;
    NSError *__autoreleasing *errorPtr = &error;
    [inv setArgument:&modelArg atIndex:2];
    [inv setArgument:&optionsArg atIndex:3];
    [inv setArgument:&qosValue atIndex:4];
    [inv setArgument:&errorPtr atIndex:5];
    @try {
        [inv invoke];
    } @catch (NSException *exception) {
        if (errorText) *errorText = [NSString stringWithFormat:@"exception: %@", exception.reason];
        return NO;
    }

    BOOL ok = NO;
    if (sig.methodReturnType[0] == 'B' || sig.methodReturnType[0] == 'c' || sig.methodReturnType[0] == 'C') {
        [inv getReturnValue:&ok];
    } else {
        if (errorText) *errorText = [NSString stringWithFormat:@"<unsupported return type %s>", sig.methodReturnType];
        return NO;
    }
    if (outError) *outError = error;
    return ok;
}

static NSDictionary *MakeRawWeightDict(NSData *blob) {
    return @{
        @"w": blob,
        @"weight.bin": blob,
        @"weights/weight.bin": blob,
        @"@model_path/weights/weight.bin": blob,
    };
}

static NSDictionary *MakeCanonicalRawWeightDict(NSData *blob) {
    return @{ @"w": blob };
}

static NSDictionary *MakeSingleKeyRawWeightDict(NSData *blob, NSString *key) {
    return @{ key: blob };
}

static NSDictionary *MakeANEWeightDict(NSData *blob, NSMutableDictionary *caseSummary) {
    Class weightClass = NSClassFromString(@"_ANEWeight");
    SEL sel = NSSelectorFromString(@"weightWithSymbolAndURLSHA:weightURL:SHACode:");
    if (!weightClass || !sel || ![(id)weightClass respondsToSelector:sel]) {
        caseSummary[@"weight_object_available"] = @NO;
        return nil;
    }

    NSString *path = @"/tmp/ane_inmemory_model_probe_weight.bin";
    [blob writeToFile:path atomically:YES];
    NSURL *url = [NSURL fileURLWithPath:path];
    NSData *sha = SHA256(blob);

    NSMethodSignature *sig = [(id)weightClass methodSignatureForSelector:sel];
    if (!sig || sig.numberOfArguments != 5 || sig.methodReturnType[0] != '@') {
        caseSummary[@"weight_object_available"] = @NO;
        caseSummary[@"weight_object_error"] = @"<unsupported signature>";
        return nil;
    }

    NSInvocation *inv = [NSInvocation invocationWithMethodSignature:sig];
    inv.target = (id)weightClass;
    inv.selector = sel;
    __unsafe_unretained id symbol = @"w";
    __unsafe_unretained id urlArg = url;
    __unsafe_unretained id shaArg = sha;
    [inv setArgument:&symbol atIndex:2];
    [inv setArgument:&urlArg atIndex:3];
    [inv setArgument:&shaArg atIndex:4];
    @try {
        [inv invoke];
    } @catch (NSException *exception) {
        caseSummary[@"weight_object_available"] = @NO;
        caseSummary[@"weight_object_error"] = [NSString stringWithFormat:@"exception: %@", exception.reason];
        return nil;
    }

    __unsafe_unretained id weight = nil;
    [inv getReturnValue:&weight];
    if (!weight) {
        caseSummary[@"weight_object_available"] = @NO;
        caseSummary[@"weight_object_error"] = @"factory returned nil";
        return nil;
    }

    caseSummary[@"weight_object_available"] = @YES;
    caseSummary[@"weight_file_path"] = path;
    caseSummary[@"weight_sha256_bytes"] = @(sha.length);
    return @{
        @"w": weight,
        @"weight.bin": weight,
        @"weights/weight.bin": weight,
        @"@model_path/weights/weight.bin": weight,
    };
}

static NSDictionary *WrapWeightDictValues(NSDictionary *weights) {
    NSMutableDictionary *wrapped = [NSMutableDictionary dictionaryWithCapacity:weights.count];
    for (NSString *key in weights) {
        wrapped[key] = @[ weights[key] ];
    }
    return wrapped;
}

static NSDictionary *NestWeightDictByPath(NSDictionary *weights) {
    return @{
        @"@model_path/weights/weight.bin": weights,
        @"weights/weight.bin": weights,
        @"weight.bin": weights,
    };
}

static NSDictionary *NestWeightDictCanonical(NSDictionary *weights) {
    return @{ @"@model_path/weights/weight.bin": weights };
}

static NSDictionary *RunCase(NSString *name,
                             NSString *descriptorSelector,
                             NSData *milText,
                             NSDictionary *weights,
                             NSString *weightValueClass,
                             NSData *optionsPlist,
                             NSDictionary *compileOptions,
                             Class descriptorClass,
                             Class modelClass) {
    id modelProto = [modelClass alloc];
    NSDictionary *effectiveCompileOptions = compileOptions;
    NSString *compilerOptionsFileNameOverride = nil;
    if ([compileOptions isKindOfClass:[NSDictionary class]]) {
        id overrideValue = compileOptions[kProbeSetCompilerOptionsFileNameKey];
        if ([overrideValue isKindOfClass:[NSString class]]) {
            compilerOptionsFileNameOverride = overrideValue;
            NSMutableDictionary *sanitizedCompileOptions = [compileOptions mutableCopy];
            [sanitizedCompileOptions removeObjectForKey:kProbeSetCompilerOptionsFileNameKey];
            effectiveCompileOptions = sanitizedCompileOptions.count ? sanitizedCompileOptions : @{};
        }
    }

    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    out[@"name"] = name;
    out[@"descriptor_selector"] = descriptorSelector;
    out[@"mil_bytes"] = @(milText.length);
    out[@"weight_keys"] = [[weights allKeys] sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];
    out[@"weight_value_class"] = weightValueClass;
    out[@"descriptor_signature"] = SignatureString((id)descriptorClass, descriptorSelector);
    out[@"init_signature"] = SignatureString(modelProto, @"initWithDesctiptor:");
    out[@"compile_signature"] = SignatureString(modelProto, @"compileWithQoS:options:error:");
    out[@"load_signature"] = SignatureString(modelProto, @"loadWithQoS:options:error:");
    out[@"compile_options_class"] = effectiveCompileOptions ? NSStringFromClass([effectiveCompileOptions class]) : @"(nil)";
    out[@"compile_options_keys"] = effectiveCompileOptions ? [[effectiveCompileOptions allKeys] sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)] : @[];
    if (compilerOptionsFileNameOverride) out[@"compiler_options_file_name_override"] = compilerOptionsFileNameOverride;

    NSString *descriptorError = nil;
    NSData *descriptorOptions = optionsPlist ?: EmptyOptionsPlist();
    out[@"options_plist_bytes"] = @(descriptorOptions.length);
    id descriptor = InvokeDescriptorFactory(descriptorClass, descriptorSelector, milText, weights, descriptorOptions, &descriptorError);
    out[@"descriptor_created"] = @(descriptor != nil);
    if (descriptorError) out[@"descriptor_error"] = descriptorError;
    if (!descriptor) return out;

    NSString *modelError = nil;
    id model = InitInMemoryModel(modelClass, descriptor, &modelError);
    out[@"model_created"] = @(model != nil);
    if (modelError) out[@"model_error"] = modelError;
    if (!model) return out;

    NSString *compilerOptionsFileNameError = nil;
    id compilerOptionsFileName = InvokeObjectNoArg(model, @"compilerOptionsFileName", &compilerOptionsFileNameError);
    if (compilerOptionsFileNameError) out[@"compiler_options_file_name_error"] = compilerOptionsFileNameError;
    if (compilerOptionsFileName) out[@"compiler_options_file_name"] = [compilerOptionsFileName description];
    if (compilerOptionsFileNameOverride) {
        NSString *setCompilerOptionsFileNameError = nil;
        BOOL setCompilerOptionsFileNameOK = InvokeVoidWithObject(model,
                                                                 @"setCompilerOptionsFileName:",
                                                                 compilerOptionsFileNameOverride,
                                                                 &setCompilerOptionsFileNameError);
        out[@"set_compiler_options_file_name_ok"] = @(setCompilerOptionsFileNameOK);
        if (setCompilerOptionsFileNameError) out[@"set_compiler_options_file_name_error"] = setCompilerOptionsFileNameError;

        NSString *compilerOptionsFileNameAfterOverrideError = nil;
        id compilerOptionsFileNameAfterOverride = InvokeObjectNoArg(model,
                                                                    @"compilerOptionsFileName",
                                                                    &compilerOptionsFileNameAfterOverrideError);
        if (compilerOptionsFileNameAfterOverrideError) out[@"compiler_options_file_name_after_override_error"] = compilerOptionsFileNameAfterOverrideError;
        if (compilerOptionsFileNameAfterOverride) out[@"compiler_options_file_name_after_override"] = [compilerOptionsFileNameAfterOverride description];
    }

    NSString *derivedCompilerOptionsError = nil;
    id derivedCompilerOptions = InvokeObjectWithObjectBool(model,
                                                           @"compilerOptionsWithOptions:isCompiledModelCached:",
                                                           effectiveCompileOptions,
                                                           NO,
                                                           &derivedCompilerOptionsError);
    if (derivedCompilerOptionsError) out[@"derived_compiler_options_error"] = derivedCompilerOptionsError;
    if (derivedCompilerOptions) {
        out[@"derived_compiler_options_class"] = NSStringFromClass([derivedCompilerOptions class]);
        out[@"derived_compiler_options_description"] = [derivedCompilerOptions description];
    }

    NSError *compileNSError = nil;
    NSString *compileText = nil;
    BOOL compiled = InvokeBoolQoS(model, @"compileWithQoS:options:error:", 0, effectiveCompileOptions, &compileNSError, &compileText);
    out[@"compile_ok"] = @(compiled);
    if (compileText) out[@"compile_error"] = compileText;
    if (compileNSError) {
        out[@"compile_nserror"] = compileNSError.localizedDescription ?: @"";
        out[@"compile_nserror_code"] = @(compileNSError.code);
        out[@"compile_nserror_domain"] = compileNSError.domain ?: @"";
    }

    NSString *pathError = nil;
    id localPath = InvokeObjectNoArg(model, @"localModelPath", &pathError);
    if (pathError) out[@"local_model_path_error"] = pathError;
    if (localPath) out[@"local_model_path"] = [localPath description];

    if (!compiled) return out;

    NSError *loadNSError = nil;
    NSString *loadText = nil;
    BOOL loaded = InvokeBoolQoS(model, @"loadWithQoS:options:error:", 0, @{}, &loadNSError, &loadText);
    out[@"load_ok"] = @(loaded);
    if (loadText) out[@"load_error"] = loadText;
    if (loadNSError) {
        out[@"load_nserror"] = loadNSError.localizedDescription ?: @"";
        out[@"load_nserror_code"] = @(loadNSError.code);
        out[@"load_nserror_domain"] = loadNSError.domain ?: @"";
    }
    return out;
}

static NSDictionary *RunDirectClientCompileCase(NSString *name,
                                                NSString *modelDirPath,
                                                Class aneModelClass,
                                                Class aneClientClass) {
    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    out[@"name"] = name;
    out[@"model_dir_path"] = modelDirPath ?: @"(nil)";
    out[@"ane_model_signature"] = SignatureString((id)aneModelClass, @"modelAtURL:key:");
    out[@"ane_client_signature"] = SignatureString((id)aneClientClass, @"compileModel:options:qos:error:");
    if (!modelDirPath.length || !aneModelClass || !aneClientClass) {
        out[@"ane_model_created"] = @NO;
        return out;
    }

    NSURL *modelURL = [NSURL fileURLWithPath:modelDirPath isDirectory:YES];
    NSString *modelKey = modelURL.lastPathComponent ?: @"in_memory_probe";
    NSString *modelError = nil;
    id aneModel = InvokeObjectWithTwoObjects((id)aneModelClass,
                                             @"modelAtURL:key:",
                                             modelURL,
                                             modelKey,
                                             &modelError);
    out[@"ane_model_created"] = @(aneModel != nil);
    if (modelError) out[@"ane_model_error"] = modelError;
    if (!aneModel) return out;

    NSString *clientError = nil;
    id client = InvokeObjectNoArg((id)aneClientClass, @"sharedConnection", &clientError);
    out[@"ane_client_created"] = @(client != nil);
    if (clientError) out[@"ane_client_error"] = clientError;
    if (!client) return out;

    NSError *compileNSError = nil;
    NSString *compileText = nil;
    BOOL compiled = InvokeBoolWithObjectObjectQoS(client,
                                                  @"compileModel:options:qos:error:",
                                                  aneModel,
                                                  @{},
                                                  0,
                                                  &compileNSError,
                                                  &compileText);
    out[@"client_compile_ok"] = @(compiled);
    if (compileText) out[@"client_compile_error"] = compileText;
    if (compileNSError) {
        out[@"client_compile_nserror"] = compileNSError.localizedDescription ?: @"";
        out[@"client_compile_nserror_code"] = @(compileNSError.code);
        out[@"client_compile_nserror_domain"] = compileNSError.domain ?: @"";
    }
    return out;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSString *summaryPath = SummaryPathFromArgs(argc, argv);
        NSMutableDictionary *summary = [NSMutableDictionary dictionary];
        summary[@"probe"] = @"ane_inmemory_model_probe";
        summary[@"framework_path"] = kAppleNeuralEnginePath;

        Log(@"=== load private framework ===");
        BOOL loaded = LoadImage(kAppleNeuralEnginePath);
        summary[@"framework_loaded"] = @(loaded);
        if (!loaded) return 1;

        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class weightClass = NSClassFromString(@"_ANEWeight");
        Class aneModelClass = NSClassFromString(@"_ANEModel");
        Class aneClientClass = NSClassFromString(@"_ANEClient");
        summary[@"classes"] = @{
            @"_ANEInMemoryModelDescriptor": @(descriptorClass != Nil),
            @"_ANEInMemoryModel": @(modelClass != Nil),
            @"_ANEWeight": @(weightClass != Nil),
            @"_ANEModel": @(aneModelClass != Nil),
            @"_ANEClient": @(aneClientClass != Nil),
        };
        if (!descriptorClass || !modelClass) {
            Log(@"missing required classes: descriptor=%@ model=%@",
                descriptorClass ? @"YES" : @"NO",
                modelClass ? @"YES" : @"NO");
            return 2;
        }

        NSMutableArray *cases = [NSMutableArray array];

        Log(@"\n=== case inline_empty ===");
        NSDictionary *inlineCase = RunCase(@"inline_empty",
                           @"modelWithMILText:weights:optionsPlist:",
                                           InlineMILText(),
                                           @{},
                                           @"(none)",
                                           nil,
                                           @{},
                                           descriptorClass,
                                           modelClass);
        [cases addObject:inlineCase];
        Log(@"  descriptor=%@ compile=%@ load=%@",
            [inlineCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
            [inlineCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
            [inlineCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
        if (inlineCase[@"compile_nserror"]) Log(@"  compile error: %@", inlineCase[@"compile_nserror"]);
        if (inlineCase[@"load_nserror"]) Log(@"  load error: %@", inlineCase[@"load_nserror"]);

        NSData *blob = WeightBlobData();
        NSData *blobMIL = BlobMILText();

        Log(@"\n=== case blob_raw_nsdata ===");
        NSDictionary *rawCase = RunCase(@"blob_raw_nsdata",
                        @"modelWithMILText:weights:optionsPlist:",
                                        blobMIL,
                                        MakeRawWeightDict(blob),
                                        @"NSData",
                                        nil,
                                        @{},
                                        descriptorClass,
                                        modelClass);
        [cases addObject:rawCase];
        Log(@"  descriptor=%@ compile=%@ load=%@",
            [rawCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
            [rawCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
            [rawCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
        if (rawCase[@"compile_nserror"]) Log(@"  compile error: %@", rawCase[@"compile_nserror"]);
        if (rawCase[@"load_nserror"]) Log(@"  load error: %@", rawCase[@"load_nserror"]);

        Log(@"\n=== case blob_aneweight ===");
        NSMutableDictionary *weightInfo = [NSMutableDictionary dictionary];
        NSDictionary *aneWeightDict = MakeANEWeightDict(blob, weightInfo);
        NSMutableDictionary *aneWeightCase = nil;
        if (!aneWeightDict) {
            aneWeightCase = [@{
                @"name": @"blob_aneweight",
                @"weight_value_class": @"_ANEWeight",
                @"descriptor_created": @NO,
            } mutableCopy];
            [aneWeightCase addEntriesFromDictionary:weightInfo];
        } else {
            aneWeightCase = [[RunCase(@"blob_aneweight",
                                      @"modelWithMILText:weights:optionsPlist:",
                                      blobMIL,
                                      aneWeightDict,
                                      @"_ANEWeight",
                                      nil,
                                      @{},
                                      descriptorClass,
                                      modelClass) mutableCopy] mutableCopy];
            [aneWeightCase addEntriesFromDictionary:weightInfo];
        }
        [cases addObject:aneWeightCase];
        Log(@"  descriptor=%@ compile=%@ load=%@",
            [aneWeightCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
            [aneWeightCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
            [aneWeightCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
        if (aneWeightCase[@"compile_nserror"]) Log(@"  compile error: %@", aneWeightCase[@"compile_nserror"]);
        if (aneWeightCase[@"load_nserror"]) Log(@"  load error: %@", aneWeightCase[@"load_nserror"]);
        if (aneWeightCase[@"weight_object_error"]) Log(@"  _ANEWeight setup: %@", aneWeightCase[@"weight_object_error"]);

        Log(@"\n=== case blob_raw_nsdata_array ===");
        NSDictionary *rawArrayCase = RunCase(@"blob_raw_nsdata_array",
                             @"modelWithMILText:weights:optionsPlist:",
                                             blobMIL,
                                             WrapWeightDictValues(MakeRawWeightDict(blob)),
                                             @"NSArray<NSData>",
                                             nil,
                                             @{},
                                             descriptorClass,
                                             modelClass);
        [cases addObject:rawArrayCase];
        Log(@"  descriptor=%@ compile=%@ load=%@",
            [rawArrayCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
            [rawArrayCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
            [rawArrayCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
        if (rawArrayCase[@"compile_nserror"]) Log(@"  compile error: %@", rawArrayCase[@"compile_nserror"]);
        if (rawArrayCase[@"load_nserror"]) Log(@"  load error: %@", rawArrayCase[@"load_nserror"]);
        if (rawArrayCase[@"descriptor_error"]) Log(@"  descriptor error: %@", rawArrayCase[@"descriptor_error"]);

        Log(@"\n=== case blob_raw_nsdata_nested ===");
        NSDictionary *rawNestedCase = RunCase(@"blob_raw_nsdata_nested",
                              @"modelWithMILText:weights:optionsPlist:",
                                              blobMIL,
                                              NestWeightDictByPath(MakeRawWeightDict(blob)),
                                              @"NSDictionary<path,NSDictionary<NSString,NSData>>",
                                              nil,
                                              @{},
                                              descriptorClass,
                                              modelClass);
        [cases addObject:rawNestedCase];
        Log(@"  descriptor=%@ compile=%@ load=%@",
            [rawNestedCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
            [rawNestedCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
            [rawNestedCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
        if (rawNestedCase[@"compile_nserror"]) Log(@"  compile error: %@", rawNestedCase[@"compile_nserror"]);
        if (rawNestedCase[@"load_nserror"]) Log(@"  load error: %@", rawNestedCase[@"load_nserror"]);
        if (rawNestedCase[@"descriptor_error"]) Log(@"  descriptor error: %@", rawNestedCase[@"descriptor_error"]);

        Log(@"\n=== case blob_raw_nsdata_nested_canonical ===");
        NSDictionary *rawNestedCanonicalCase = RunCase(@"blob_raw_nsdata_nested_canonical",
                                       @"modelWithMILText:weights:optionsPlist:",
                                                       blobMIL,
                                                       NestWeightDictCanonical(MakeCanonicalRawWeightDict(blob)),
                                                       @"NSDictionary<@model_path/weights/weight.bin,NSDictionary<w,NSData>>",
                                                       nil,
                                                       @{},
                                                       descriptorClass,
                                                       modelClass);
        [cases addObject:rawNestedCanonicalCase];
        Log(@"  descriptor=%@ compile=%@ load=%@",
            [rawNestedCanonicalCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
            [rawNestedCanonicalCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
            [rawNestedCanonicalCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
        if (rawNestedCanonicalCase[@"compile_nserror"]) Log(@"  compile error: %@", rawNestedCanonicalCase[@"compile_nserror"]);
        if (rawNestedCanonicalCase[@"load_nserror"]) Log(@"  load error: %@", rawNestedCanonicalCase[@"load_nserror"]);
        if (rawNestedCanonicalCase[@"descriptor_error"]) Log(@"  descriptor error: %@", rawNestedCanonicalCase[@"descriptor_error"]);

        NSData *blobMILModelPathW = RewriteMILWeightPath(blobMIL, @"@model_path/weights/weight.bin", @"@model_path/w");
        if (blobMILModelPathW) {
            Log(@"\n=== case blob_raw_nsdata_nested_canonical_modelpath_w ===");
            NSDictionary *rawNestedCanonicalModelPathWCase = RunCase(@"blob_raw_nsdata_nested_canonical_modelpath_w",
                                                     @"modelWithMILText:weights:optionsPlist:",
                                                                     blobMILModelPathW,
                                                                     NestWeightDictCanonical(MakeCanonicalRawWeightDict(blob)),
                                                                     @"NSDictionary<@model_path/weights/weight.bin,NSDictionary<w,NSData>> + MIL:@model_path/w",
                                                                     nil,
                                                                     @{},
                                                                     descriptorClass,
                                                                     modelClass);
            [cases addObject:rawNestedCanonicalModelPathWCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [rawNestedCanonicalModelPathWCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [rawNestedCanonicalModelPathWCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [rawNestedCanonicalModelPathWCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (rawNestedCanonicalModelPathWCase[@"compile_nserror"]) Log(@"  compile error: %@", rawNestedCanonicalModelPathWCase[@"compile_nserror"]);
            if (rawNestedCanonicalModelPathWCase[@"load_nserror"]) Log(@"  load error: %@", rawNestedCanonicalModelPathWCase[@"load_nserror"]);
            if (rawNestedCanonicalModelPathWCase[@"descriptor_error"]) Log(@"  descriptor error: %@", rawNestedCanonicalModelPathWCase[@"descriptor_error"]);
        }

        if (aneWeightDict) {
            Log(@"\n=== case blob_aneweight_array ===");
            NSDictionary *aneWeightArrayCase = RunCase(@"blob_aneweight_array",
                                                       @"modelWithMILText:weights:optionsPlist:",
                                                       blobMIL,
                                                       WrapWeightDictValues(aneWeightDict),
                                                       @"NSArray<_ANEWeight>",
                                                       nil,
                                                       @{},
                                                       descriptorClass,
                                                       modelClass);
            NSMutableDictionary *aneWeightArrayCaseMutable = [aneWeightArrayCase mutableCopy];
            [aneWeightArrayCaseMutable addEntriesFromDictionary:weightInfo];
            [cases addObject:aneWeightArrayCaseMutable];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [aneWeightArrayCaseMutable[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [aneWeightArrayCaseMutable[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [aneWeightArrayCaseMutable[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (aneWeightArrayCaseMutable[@"compile_nserror"]) Log(@"  compile error: %@", aneWeightArrayCaseMutable[@"compile_nserror"]);
            if (aneWeightArrayCaseMutable[@"load_nserror"]) Log(@"  load error: %@", aneWeightArrayCaseMutable[@"load_nserror"]);
            if (aneWeightArrayCaseMutable[@"descriptor_error"]) Log(@"  descriptor error: %@", aneWeightArrayCaseMutable[@"descriptor_error"]);

            Log(@"\n=== case blob_aneweight_nested ===");
            NSDictionary *aneWeightNestedCase = RunCase(@"blob_aneweight_nested",
                                                        @"modelWithMILText:weights:optionsPlist:",
                                                        blobMIL,
                                                        NestWeightDictByPath(aneWeightDict),
                                                        @"NSDictionary<path,NSDictionary<NSString,_ANEWeight>>",
                                                        nil,
                                                        @{},
                                                        descriptorClass,
                                                        modelClass);
            NSMutableDictionary *aneWeightNestedCaseMutable = [aneWeightNestedCase mutableCopy];
            [aneWeightNestedCaseMutable addEntriesFromDictionary:weightInfo];
            [cases addObject:aneWeightNestedCaseMutable];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [aneWeightNestedCaseMutable[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [aneWeightNestedCaseMutable[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [aneWeightNestedCaseMutable[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (aneWeightNestedCaseMutable[@"compile_nserror"]) Log(@"  compile error: %@", aneWeightNestedCaseMutable[@"compile_nserror"]);
            if (aneWeightNestedCaseMutable[@"load_nserror"]) Log(@"  load error: %@", aneWeightNestedCaseMutable[@"load_nserror"]);
            if (aneWeightNestedCaseMutable[@"descriptor_error"]) Log(@"  descriptor error: %@", aneWeightNestedCaseMutable[@"descriptor_error"]);
        }

        Log(@"\n=== case replay_qwen_b1_raw_nested ===");
        NSMutableDictionary *replayCase = [NSMutableDictionary dictionary];
        NSData *replayMIL = ReadFileData(kReplayMILPath, replayCase, @"mil_source");
        NSData *replayWeight = ReadFileData(kReplayWeightPath, replayCase, @"weight_source");
        NSData *replayCoreMLData = ReadFileData(kReplayCoreMLDataPath, replayCase, @"coremldata_source");
        NSData *replayAnalyticsCoreMLData = ReadFileData(kReplayAnalyticsCoreMLDataPath, replayCase, @"analytics_coremldata_source");
        if (!replayMIL || !replayWeight || !replayCoreMLData || !replayAnalyticsCoreMLData) {
            replayCase[@"name"] = @"replay_qwen_b1_raw_nested";
            replayCase[@"descriptor_created"] = @NO;
            replayCase[@"weight_value_class"] = @"NSDictionary<path,NSDictionary<NSString,NSData>>";
            [cases addObject:replayCase];
            Log(@"  descriptor=NO compile=NO load=NO");
            if (replayCase[@"mil_source_error"]) Log(@"  MIL read error: %@", replayCase[@"mil_source_error"]);
            if (replayCase[@"weight_source_error"]) Log(@"  weight read error: %@", replayCase[@"weight_source_error"]);
            if (replayCase[@"coremldata_source_error"]) Log(@"  coremldata read error: %@", replayCase[@"coremldata_source_error"]);
            if (replayCase[@"analytics_coremldata_source_error"]) Log(@"  analytics coremldata read error: %@", replayCase[@"analytics_coremldata_source_error"]);
        } else {
            NSString *replayModelDirPath = [kReplayMILPath stringByDeletingLastPathComponent];
            Log(@"\n=== case replay_qwen_b1_original_mlmodelc_direct_client_compile ===");
            NSDictionary *replayOriginalDirectCompileCase = RunDirectClientCompileCase(@"replay_qwen_b1_original_mlmodelc_direct_client_compile",
                                                                                        replayModelDirPath,
                                                                                        aneModelClass,
                                                                                        aneClientClass);
            [cases addObject:replayOriginalDirectCompileCase];
            Log(@"  ane_model=%@ client=%@ direct_compile=%@",
                [replayOriginalDirectCompileCase[@"ane_model_created"] boolValue] ? @"YES" : @"NO",
                [replayOriginalDirectCompileCase[@"ane_client_created"] boolValue] ? @"YES" : @"NO",
                [replayOriginalDirectCompileCase[@"client_compile_ok"] boolValue] ? @"YES" : @"NO");
            if (replayOriginalDirectCompileCase[@"client_compile_nserror"]) Log(@"  direct compile error: %@", replayOriginalDirectCompileCase[@"client_compile_nserror"]);
            if (replayOriginalDirectCompileCase[@"ane_model_error"]) Log(@"  model setup error: %@", replayOriginalDirectCompileCase[@"ane_model_error"]);
            if (replayOriginalDirectCompileCase[@"ane_client_error"]) Log(@"  client setup error: %@", replayOriginalDirectCompileCase[@"ane_client_error"]);

            NSDictionary *replayWeights = NestWeightDictByPath(MakeRawWeightDict(replayWeight));
            NSDictionary *replayCanonicalWeights = NestWeightDictCanonical(MakeCanonicalRawWeightDict(replayWeight));
            NSData *replayMILModelPathW = RewriteMILWeightPath(replayMIL, @"@model_path/weights/weight.bin", @"@model_path/w");
            NSMutableDictionary *replayRunCase = [[RunCase(@"replay_qwen_b1_raw_nested",
                                                           @"modelWithMILText:weights:optionsPlist:",
                                                           replayMIL,
                                                           replayWeights,
                                                           @"NSDictionary<path,NSDictionary<NSString,NSData>>",
                                                           nil,
                                                           @{},
                                                           descriptorClass,
                                                           modelClass) mutableCopy] mutableCopy];
            [replayRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replayRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replayRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replayRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replayRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replayRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayRunCase[@"compile_nserror"]);
            if (replayRunCase[@"load_nserror"]) Log(@"  load error: %@", replayRunCase[@"load_nserror"]);
            if (replayRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayRunCase[@"descriptor_error"]);

            Log(@"\n=== case replay_qwen_b1_raw_nested_canonical ===");
            NSMutableDictionary *replayCanonicalRunCase = [[RunCase(@"replay_qwen_b1_raw_nested_canonical",
                                                                    @"modelWithMILText:weights:optionsPlist:",
                                                                    replayMIL,
                                                                    replayCanonicalWeights,
                                                                    @"NSDictionary<@model_path/weights/weight.bin,NSDictionary<w,NSData>>",
                                                                    nil,
                                                                    @{},
                                                                    descriptorClass,
                                                                    modelClass) mutableCopy] mutableCopy];
            [replayCanonicalRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replayCanonicalRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replayCanonicalRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replayCanonicalRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replayCanonicalRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replayCanonicalRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayCanonicalRunCase[@"compile_nserror"]);
            if (replayCanonicalRunCase[@"load_nserror"]) Log(@"  load error: %@", replayCanonicalRunCase[@"load_nserror"]);
            if (replayCanonicalRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayCanonicalRunCase[@"descriptor_error"]);

            if (replayMILModelPathW) {
                Log(@"\n=== case replay_qwen_b1_raw_nested_modelpath_w ===");
                NSMutableDictionary *replayModelPathWRunCase = [[RunCase(@"replay_qwen_b1_raw_nested_modelpath_w",
                                                                         @"modelWithMILText:weights:optionsPlist:",
                                                                         replayMILModelPathW,
                                                                         replayCanonicalWeights,
                                                                         @"NSDictionary<@model_path/weights/weight.bin,NSDictionary<w,NSData>> + MIL:@model_path/w",
                                                                         nil,
                                                                         @{},
                                                                         descriptorClass,
                                                                         modelClass) mutableCopy] mutableCopy];
                [replayModelPathWRunCase addEntriesFromDictionary:replayCase];
                [cases addObject:replayModelPathWRunCase];
                Log(@"  descriptor=%@ compile=%@ load=%@",
                    [replayModelPathWRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                    [replayModelPathWRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                    [replayModelPathWRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
                if (replayModelPathWRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayModelPathWRunCase[@"compile_nserror"]);
                if (replayModelPathWRunCase[@"load_nserror"]) Log(@"  load error: %@", replayModelPathWRunCase[@"load_nserror"]);
                if (replayModelPathWRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayModelPathWRunCase[@"descriptor_error"]);

                NSString *replayModelPathWLocalModelPath = replayModelPathWRunCase[@"local_model_path"];
                if (replayModelPathWLocalModelPath.length > 0) {
                    Log(@"\n=== case replay_qwen_b1_raw_nested_modelpath_w_direct_client_compile ===");
                    NSDictionary *replayModelPathWDirectCompileCase = RunDirectClientCompileCase(@"replay_qwen_b1_raw_nested_modelpath_w_direct_client_compile",
                                                                                                  replayModelPathWLocalModelPath,
                                                                                                  aneModelClass,
                                                                                                  aneClientClass);
                    [cases addObject:replayModelPathWDirectCompileCase];
                    Log(@"  ane_model=%@ client=%@ direct_compile=%@",
                        [replayModelPathWDirectCompileCase[@"ane_model_created"] boolValue] ? @"YES" : @"NO",
                        [replayModelPathWDirectCompileCase[@"ane_client_created"] boolValue] ? @"YES" : @"NO",
                        [replayModelPathWDirectCompileCase[@"client_compile_ok"] boolValue] ? @"YES" : @"NO");
                    if (replayModelPathWDirectCompileCase[@"client_compile_nserror"]) Log(@"  direct compile error: %@", replayModelPathWDirectCompileCase[@"client_compile_nserror"]);
                    if (replayModelPathWDirectCompileCase[@"ane_model_error"]) Log(@"  model setup error: %@", replayModelPathWDirectCompileCase[@"ane_model_error"]);
                    if (replayModelPathWDirectCompileCase[@"ane_client_error"]) Log(@"  client setup error: %@", replayModelPathWDirectCompileCase[@"ane_client_error"]);
                }
            }

            Log(@"\n=== case replay_qwen_b1_raw_nested_relpath_key ===");
            NSMutableDictionary *replayRelpathKeyRunCase = [[RunCase(@"replay_qwen_b1_raw_nested_relpath_key",
                                                                     @"modelWithMILText:weights:optionsPlist:",
                                                                     replayMIL,
                                                                     NestWeightDictCanonical(MakeSingleKeyRawWeightDict(replayWeight, @"weights/weight.bin")),
                                                                     @"NSDictionary<@model_path/weights/weight.bin,NSDictionary<weights/weight.bin,NSData>>",
                                                                     nil,
                                                                     @{},
                                                                     descriptorClass,
                                                                     modelClass) mutableCopy] mutableCopy];
            [replayRelpathKeyRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replayRelpathKeyRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replayRelpathKeyRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replayRelpathKeyRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replayRelpathKeyRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replayRelpathKeyRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayRelpathKeyRunCase[@"compile_nserror"]);
            if (replayRelpathKeyRunCase[@"load_nserror"]) Log(@"  load error: %@", replayRelpathKeyRunCase[@"load_nserror"]);
            if (replayRelpathKeyRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayRelpathKeyRunCase[@"descriptor_error"]);

            Log(@"\n=== case replay_qwen_b1_raw_nested_coremldata ===");
            NSMutableDictionary *replayCoreMLRunCase = [[RunCase(@"replay_qwen_b1_raw_nested_coremldata",
                                                                 @"modelWithMILText:weights:optionsPlist:",
                                                                 replayMIL,
                                                                 replayWeights,
                                                                 @"NSDictionary<path,NSDictionary<NSString,NSData>>",
                                                                 replayCoreMLData,
                                                                 @{},
                                                                 descriptorClass,
                                                                 modelClass) mutableCopy] mutableCopy];
            [replayCoreMLRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replayCoreMLRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replayCoreMLRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replayCoreMLRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replayCoreMLRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replayCoreMLRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayCoreMLRunCase[@"compile_nserror"]);
            if (replayCoreMLRunCase[@"load_nserror"]) Log(@"  load error: %@", replayCoreMLRunCase[@"load_nserror"]);
            if (replayCoreMLRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayCoreMLRunCase[@"descriptor_error"]);

            Log(@"\n=== case replay_qwen_b1_raw_nested_analytics_coremldata ===");
            NSMutableDictionary *replayAnalyticsCoreMLRunCase = [[RunCase(@"replay_qwen_b1_raw_nested_analytics_coremldata",
                                                                          @"modelWithMILText:weights:optionsPlist:",
                                                                          replayMIL,
                                                                          replayWeights,
                                                                          @"NSDictionary<path,NSDictionary<NSString,NSData>>",
                                                                          replayAnalyticsCoreMLData,
                                                                          @{},
                                                                          descriptorClass,
                                                                          modelClass) mutableCopy] mutableCopy];
            [replayAnalyticsCoreMLRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replayAnalyticsCoreMLRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replayAnalyticsCoreMLRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replayAnalyticsCoreMLRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replayAnalyticsCoreMLRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replayAnalyticsCoreMLRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayAnalyticsCoreMLRunCase[@"compile_nserror"]);
            if (replayAnalyticsCoreMLRunCase[@"load_nserror"]) Log(@"  load error: %@", replayAnalyticsCoreMLRunCase[@"load_nserror"]);
            if (replayAnalyticsCoreMLRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayAnalyticsCoreMLRunCase[@"descriptor_error"]);

            Log(@"\n=== case replay_qwen_b1_networkdesc_coremldata ===");
            NSMutableDictionary *replayNetworkDescRunCase = [[RunCase(@"replay_qwen_b1_networkdesc_coremldata",
                                                                      @"modelWithNetworkDescription:weights:optionsPlist:",
                                                                      replayCoreMLData,
                                                                      replayWeights,
                                                                      @"NSDictionary<path,NSDictionary<NSString,NSData>>",
                                                                      nil,
                                                                      @{},
                                                                      descriptorClass,
                                                                      modelClass) mutableCopy] mutableCopy];
            [replayNetworkDescRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replayNetworkDescRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replayNetworkDescRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replayNetworkDescRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replayNetworkDescRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replayNetworkDescRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayNetworkDescRunCase[@"compile_nserror"]);
            if (replayNetworkDescRunCase[@"load_nserror"]) Log(@"  load error: %@", replayNetworkDescRunCase[@"load_nserror"]);
            if (replayNetworkDescRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayNetworkDescRunCase[@"descriptor_error"]);

            Log(@"\n=== case replay_qwen_b1_raw_nested_compileopts_nil ===");
            NSMutableDictionary *replayNilCompileOptionsRunCase = [[RunCase(@"replay_qwen_b1_raw_nested_compileopts_nil",
                                                                            @"modelWithMILText:weights:optionsPlist:",
                                                                            replayMIL,
                                                                            replayWeights,
                                                                            @"NSDictionary<path,NSDictionary<NSString,NSData>>",
                                                                            nil,
                                                                            nil,
                                                                            descriptorClass,
                                                                            modelClass) mutableCopy] mutableCopy];
            [replayNilCompileOptionsRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replayNilCompileOptionsRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replayNilCompileOptionsRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replayNilCompileOptionsRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replayNilCompileOptionsRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replayNilCompileOptionsRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayNilCompileOptionsRunCase[@"compile_nserror"]);
            if (replayNilCompileOptionsRunCase[@"load_nserror"]) Log(@"  load error: %@", replayNilCompileOptionsRunCase[@"load_nserror"]);
            if (replayNilCompileOptionsRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayNilCompileOptionsRunCase[@"descriptor_error"]);

            Log(@"\n=== case replay_qwen_b1_raw_nested_compileopts_probe ===");
            NSMutableDictionary *replayProbeCompileOptionsRunCase = [[RunCase(@"replay_qwen_b1_raw_nested_compileopts_probe",
                                                                              @"modelWithMILText:weights:optionsPlist:",
                                                                              replayMIL,
                                                                              replayWeights,
                                                                              @"NSDictionary<path,NSDictionary<NSString,NSData>>",
                                                                              nil,
                                                                              @{ @"copilot_probe_key": @"copilot_probe_value" },
                                                                              descriptorClass,
                                                                              modelClass) mutableCopy] mutableCopy];
            [replayProbeCompileOptionsRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replayProbeCompileOptionsRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replayProbeCompileOptionsRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replayProbeCompileOptionsRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replayProbeCompileOptionsRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replayProbeCompileOptionsRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayProbeCompileOptionsRunCase[@"compile_nserror"]);
            if (replayProbeCompileOptionsRunCase[@"load_nserror"]) Log(@"  load error: %@", replayProbeCompileOptionsRunCase[@"load_nserror"]);
            if (replayProbeCompileOptionsRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayProbeCompileOptionsRunCase[@"descriptor_error"]);

            Log(@"\n=== case replay_qwen_b1_raw_nested_compileopts_force_anecir ===");
            NSMutableDictionary *replayForceANECIRRunCase = [[RunCase(@"replay_qwen_b1_raw_nested_compileopts_force_anecir",
                                                                      @"modelWithMILText:weights:optionsPlist:",
                                                                      replayMIL,
                                                                      replayWeights,
                                                                      @"NSDictionary<path,NSDictionary<NSString,NSData>>",
                                                                      nil,
                                                                      @{ @"kANEFModelType": @"kANEFModelANECIR" },
                                                                      descriptorClass,
                                                                      modelClass) mutableCopy] mutableCopy];
            [replayForceANECIRRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replayForceANECIRRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replayForceANECIRRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replayForceANECIRRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replayForceANECIRRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replayForceANECIRRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayForceANECIRRunCase[@"compile_nserror"]);
            if (replayForceANECIRRunCase[@"load_nserror"]) Log(@"  load error: %@", replayForceANECIRRunCase[@"load_nserror"]);
            if (replayForceANECIRRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayForceANECIRRunCase[@"descriptor_error"]);

            Log(@"\n=== case replay_qwen_b1_raw_nested_compileopts_alt_filename ===");
            NSMutableDictionary *replayAltFilenameRunCase = [[RunCase(@"replay_qwen_b1_raw_nested_compileopts_alt_filename",
                                                                      @"modelWithMILText:weights:optionsPlist:",
                                                                      replayMIL,
                                                                      replayWeights,
                                                                      @"NSDictionary<path,NSDictionary<NSString,NSData>>",
                                                                      nil,
                                                                      @{ @"kANEFCompilerOptionsFilenameKey": @"copilot_missing_compiler_options.plist" },
                                                                      descriptorClass,
                                                                      modelClass) mutableCopy] mutableCopy];
            [replayAltFilenameRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replayAltFilenameRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replayAltFilenameRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replayAltFilenameRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replayAltFilenameRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replayAltFilenameRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayAltFilenameRunCase[@"compile_nserror"]);
            if (replayAltFilenameRunCase[@"load_nserror"]) Log(@"  load error: %@", replayAltFilenameRunCase[@"load_nserror"]);
            if (replayAltFilenameRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayAltFilenameRunCase[@"descriptor_error"]);

            Log(@"\n=== case replay_qwen_b1_raw_nested_setter_alt_filename ===");
            NSMutableDictionary *replaySetterAltFilenameRunCase = [[RunCase(@"replay_qwen_b1_raw_nested_setter_alt_filename",
                                                                            @"modelWithMILText:weights:optionsPlist:",
                                                                            replayMIL,
                                                                            replayWeights,
                                                                            @"NSDictionary<path,NSDictionary<NSString,NSData>>",
                                                                            nil,
                                                                            @{ kProbeSetCompilerOptionsFileNameKey: @"copilot_missing_compiler_options.plist" },
                                                                            descriptorClass,
                                                                            modelClass) mutableCopy] mutableCopy];
            [replaySetterAltFilenameRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replaySetterAltFilenameRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replaySetterAltFilenameRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replaySetterAltFilenameRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replaySetterAltFilenameRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replaySetterAltFilenameRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replaySetterAltFilenameRunCase[@"compile_nserror"]);
            if (replaySetterAltFilenameRunCase[@"load_nserror"]) Log(@"  load error: %@", replaySetterAltFilenameRunCase[@"load_nserror"]);
            if (replaySetterAltFilenameRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replaySetterAltFilenameRunCase[@"descriptor_error"]);

            Log(@"\n=== case replay_qwen_b1_raw_nested_compileopts_tiny_maxmem ===");
            NSMutableDictionary *replayTinyMaxMemRunCase = [[RunCase(@"replay_qwen_b1_raw_nested_compileopts_tiny_maxmem",
                                                                     @"modelWithMILText:weights:optionsPlist:",
                                                                     replayMIL,
                                                                     replayWeights,
                                                                     @"NSDictionary<path,NSDictionary<NSString,NSData>>",
                                                                     nil,
                                                                     @{ @"maxModelMemorySize": @(0x1000) },
                                                                     descriptorClass,
                                                                     modelClass) mutableCopy] mutableCopy];
            [replayTinyMaxMemRunCase addEntriesFromDictionary:replayCase];
            [cases addObject:replayTinyMaxMemRunCase];
            Log(@"  descriptor=%@ compile=%@ load=%@",
                [replayTinyMaxMemRunCase[@"descriptor_created"] boolValue] ? @"YES" : @"NO",
                [replayTinyMaxMemRunCase[@"compile_ok"] boolValue] ? @"YES" : @"NO",
                [replayTinyMaxMemRunCase[@"load_ok"] boolValue] ? @"YES" : @"NO");
            if (replayTinyMaxMemRunCase[@"compile_nserror"]) Log(@"  compile error: %@", replayTinyMaxMemRunCase[@"compile_nserror"]);
            if (replayTinyMaxMemRunCase[@"load_nserror"]) Log(@"  load error: %@", replayTinyMaxMemRunCase[@"load_nserror"]);
            if (replayTinyMaxMemRunCase[@"descriptor_error"]) Log(@"  descriptor error: %@", replayTinyMaxMemRunCase[@"descriptor_error"]);
        }

        summary[@"cases"] = cases;

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