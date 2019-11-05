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
/// DeviceFunctions
/// an anchor for device function extensions
public protocol DeviceFunctions {
    /// all
    func all<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Element == Bool
    /// any
    func any<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Element == Bool
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    /// exp
    func exp<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// fill(result:with:
    func fill<T>(_ result: inout T, with value: T.Element) where T: TensorView
    /// fillWithIndex(x:startAt:
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T: TensorView, T.Element: AnyNumeric
    /// neg
    /// returns the element-wise negation of `x`
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Element: SignedNumeric
    /// notEqual
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    /// reduce
    /// Reduces `x` along the specified axes
    ///
    /// - Parameter x: value tensor
    /// - Parameter into result: the scalar tensor where the result will be written
    /// - Parameter initialResult: the initial value of the result
    /// - Parameter along axes: the axes to operate on
    /// - Parameter opNext: the operation to perform on pairs of elements
    /// - Parameter opFinal: the operation to perform on the final result
    /// - Parameter body: closure that performs reduction
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func reduce<T>(x: T,
                   into result: inout T,
                   initialResult: T.Element,
                   along axes: [Int]?,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: @escaping (T.Element) -> T.Element)
        where T: TensorView
}

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
public extension DeviceFunctions {
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
            try! x.values().reduce(to: &result, initialResult, opNext)
            let buffer = try! result.readWrite()
            buffer[0] = opFinal(buffer[0])
        }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
// @Target(type:"CPU", appliedTo:"CpuAsynchronousQueue", protocols:[DeviceFunctions])
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
            queue(#function, { try (x.values(), axes) }, &result)
            { params, result in
                // TODO
            }
        } else {
            queue(#function, { try x.values() }, &result) {
                $0.reduce(to: &$1, initialResult, opNext)
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
//public extension DeviceFunctions {
//
//}
//
////******************************************************************************
//// >>>>>> GENERATED <<<<<<
//// @Target(type:"CPU", appliedTo:"CpuAsynchronousQueue", protocols:[DeviceFunctions])
//// target generated from Intent by the compiler
//#if canImport(CpuAsync)
//public extension CpuAsynchronousQueue {
//
//}
//#endif
