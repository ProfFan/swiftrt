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
import Numerics

//==============================================================================
// DeviceFunctions
public protocol DeviceFunctions: DeviceQueueBase {
    //--------------------------------------------------------------------------
    // operator functions
    /// Adds two tensors and produces their sum.
    func add<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    /// cast
    func cast<T, U>(from other: T, to result: inout U) where
        T: TensorView, T.Element: AnyConvertable,
        U: TensorView, U.Element: AnyConvertable
    /// concat
    func concat<T>(tensors: [T], alongAxis axis: Int, result: inout T) where
        T: TensorView
    /// div
    func div<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: FloatingPoint
    /// elementsAlmostEqual
    func elementsAlmostEqual<T>(lhs: T, rhs: T, tolerance: T.Element,
                                result: inout T.BoolView) where
        T: TensorView, T.Element: SignedNumeric & Comparable
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    /// exp
    func exp<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Real
    /// fill(result:with:
    func fill<T>(_ result: inout T, with value: T.Element) where T: TensorView
    /// fillWithIndex(x:startAt:
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T: TensorView, T.Element: AnyNumeric
    /// log
    func log<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Real
    /// Computes the element-wise maximum of two tensors.
    func maximum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    /// Computes the element-wise minimum of two tensors.
    func minimum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    /// mul
    func mul<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    /// neg
    /// returns the element-wise negation of `x`
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Element: SignedNumeric
    /// notEqual
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    /// pow
    func pow<T>(x: T, y: T, result: inout T) where
        T: TensorView, T.Element: Real
    /// subtract
    func subtract<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    /// sqrt
    func sqrt<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Real
    /// squared
    func squared<T>(x: T, result: inout T)
        where T: TensorView, T.Element: Numeric
    
    /// reduce
    /// Reduces `x` along the specified axes
    /// - Parameter x: value tensor
    /// - Parameter into result: the scalar tensor where the result will
    ///  be written. Dimensions with extent of 1 will be reduced
    /// - Parameter initialResult: the initial value of the result
    /// - Parameter opNext: the operation to perform on pairs of elements
    /// - Parameter opFinal: the operation to perform on the final result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func reduce<T>(x: T,
                   into result: inout T,
                   initialResult: T.Element,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: ReduceOpFinal<T>?)
        where T: TensorView
}
