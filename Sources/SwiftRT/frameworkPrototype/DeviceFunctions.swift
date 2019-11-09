//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Foundation

//==============================================================================
/// NanPropagation
public enum NanPropagation: Int, Codable {
    case propagate, noPropagate
}

//==============================================================================
/// ReductionOp
public enum ReductionOp: Int, Codable {
    case add
    case mean
    case mul
    case min
    case max
    case amax
    case asum
    case sqrtSumSquares
    case mulNonZeros
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    // reduce
    func reduce<T>(x: T,
                   into result: inout T,
                   initialResult: T.Element,
                   along axes: [Int]?,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: @escaping (T.Element) -> T.Element)
        where T: TensorView
    {
        if let axes = axes, axes.count > 0 {
            assert(axes.count <= x.rank, "rank mismatch")
            // TODO
        } else {
            do {
                x.reduce(into: &result, initialResult, opNext)
                let buffer = try result.readWrite()
                buffer[0] = opFinal(buffer[0])
            } catch {
                device.report(error)
            }
        }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
// @Target(type:"CPU", appliedTo:"CpuAsynchronousQueue", protocols:[DeviceQueue])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    // reduce
    func reduce<T>(x: T,
                   into result: inout T,
                   initialResult: T.Element,
                   along axes: [Int]?,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: @escaping (T.Element) -> T.Element)
        where T: TensorView
    {
        if let axes = axes, axes.count > 0 {
            assert(axes.count <= x.rank, "rank mismatch")
            queue(#function, { (x.values(), axes) }, &result)
            { params, result in
                // TODO
            }
        } else {
            queue(#function, { x.values() }, &result) {
                $0.reduce(into: &$1, initialResult, opNext)
                $1[$1.startIndex] = opFinal($1[$1.startIndex])
            }
        }
    }
}
#endif


////==============================================================================
//// >>>>>> User API <<<<<<
//
////------------------------------------------------------------------------------
//// >>>>>> INTENT <<<<<<
//// User device function
//public extension DeviceQueue {
//
//}
//
////******************************************************************************
//// >>>>>> GENERATED <<<<<<
//// @Target(type:"CPU", appliedTo:"CpuAsynchronousQueue", protocols:[DeviceQueue])
//// target generated from Intent by the compiler
//#if canImport(CpuAsync)
//public extension CpuAsynchronousQueue {
//
//}
//#endif
