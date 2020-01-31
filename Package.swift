// swift-tools-version:5.1
// The swift-tools-version declares the minimum version
// of Swift required to build this package.
import PackageDescription
import Foundation

#if os(Linux)
import Glibc
#else
import Darwin.C
#endif

//---------------------------------------
// test for enabled components
func isEnabled(_ id: String) -> Bool { getenv(id) != nil }
let enableAll = isEnabled("SWIFTRT_ENABLE_ALL_SERVICES")
let disableTesting = isEnabled("SWIFTRT_DISABLE_TESTING")

// Apple Cuda support has been offically dropped
#if os(Linux)
let enableCuda = enableAll || isEnabled("SWIFTRT_ENABLE_CUDA")
#else
let enableCuda = false
#endif

// if using cuda then the default is an async cpu
let enableCpuAsync = enableAll || !disableTesting ||
    isEnabled("SWIFTRT_ENABLE_ASYNC_CPU") || enableCuda

// synchronous CPU is the default case
//let enableCpuSync = enableAll || isEnabled("SWIFTRT_ENABLE_SYNC_CPU") ||
//    !enableCpuAsync
let enableCpuSync = true

// discreet asynchronous CPU for unit testing
let enableTestCpu = !disableTesting

//---------------------------------------
@available(macOS 10.13, *)
func runMakefile(target: String, workingDir: String) {
    let fileManager = FileManager()
    let task = Process()
    task.currentDirectoryURL = URL(fileURLWithPath:
    "\(fileManager.currentDirectoryPath)/\(workingDir)", isDirectory: true)
    task.executableURL = URL(fileURLWithPath: "/usr/bin/make")
    
    task.arguments = ["TARGET=\"\(target)\""]
    do {
        let outputPipe = Pipe()
        task.standardOutput = outputPipe
        try task.run()
        let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
        task.waitUntilExit()
        if task.terminationStatus != 0 {
            let output = String(decoding: outputData, as: UTF8.self)
            print(output)
        }
    } catch {
        print(error)
    }
}

//---------------------------------------
// the base products, dependencies, and targets
var products: [PackageDescription.Product] = [
    .library(name: "SwiftRT", type: .dynamic, targets: ["SwiftRT"]),
    // .library(name: "SwiftRT", type: .static, targets: ["SwiftRT"])
]
var dependencies: [Target.Dependency] = ["Numerics"]
var exclusions: [String] = []
var targets: [PackageDescription.Target] = []

//---------------------------------------
// include the cpu asynchronous service module
if enableCpuSync {
    products.append(.library(name: "CpuSync", targets: ["CpuSync"]))
    dependencies.append("CpuSync")
    targets.append(
        .systemLibrary(name: "CpuSync",
                       path: "Modules/CpuSync"))
}

//---------------------------------------
// include the cpu asynchronous service module
if enableCpuAsync {
    products.append(.library(name: "CpuAsync", targets: ["CpuAsync"]))
    dependencies.append("CpuAsync")
    targets.append(
        .systemLibrary(name: "CpuAsync",
                       path: "Modules/CpuAsync"))
}

//---------------------------------------
// include the cpu asynchronous discreet test service module
if !disableTesting {
    products.append(.library(name: "CpuTest", targets: ["CpuTest"]))
    dependencies.append("CpuTest")
    targets.append(
        .systemLibrary(name: "CpuTest",
                       path: "Modules/CpuTest"))
}

//==============================================================================
// include the Cuda service module
if enableCuda {
    //---------------------------------------
    // build kernels library
    if #available(macOS 10.13, *) {
//        runMakefile(target: ".build/debug/SwiftRTCudaKernels",
//                    workingDir: "Sources/SwiftRT/device/cuda/kernels")
    } else {
        print("OS version error. blerg...")
    }
    
    //---------------------------------------
    // add cuda components
    products.append(.library(name: "CCuda", targets: ["CCuda"]))
    dependencies.append("CCuda")
    targets.append(
        .systemLibrary(name: "CCuda",
                       path: "Modules/Cuda",
                       pkgConfig: "cuda"))
}

//---------------------------------------
// excluded unused component code
if !enableCpuSync  { exclusions.append("device/cpu/sync") }
if !enableCpuAsync { exclusions.append("device/cpu/async") }
if !enableTestCpu  { exclusions.append("device/cpu/test") }
if !enableCuda     { exclusions.append("device/cuda") }
//print("exclusions: \(exclusions)")

targets.append(
    .target(name: "SwiftRT",
            dependencies: dependencies,
            exclude: exclusions))

if !disableTesting {
    targets.append(
        .testTarget(name: "SwiftRTTests",
                    dependencies: ["SwiftRT"]))
}

//==============================================================================
// package specification
let package = Package(
    name: "SwiftRT",
    products: products,
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics",
                 .branch("master"))
    ],
    targets: targets
)
