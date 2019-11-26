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
import Real

//==============================================================================
/// cast(from:to:
/// casts elements of `x` to the output type
///
/// with placement
/// - Parameter other: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func cast<T, U>(_ other: U) -> T where
    T: TensorView, T.Element: AnyConvertable,
    U: TensorView, U.Element: AnyConvertable
{
    let name = String(describing: T.self)
    let array = TensorArray<T.Element>(count: other.elementCount, name: name)
    var result = T(shape: other.shape.dense, tensorArray: array,
                   viewOffset: 0, isShared: false)

    DeviceContext.currentQueue.cast(from: other, to: &result)
    return result
}

//==============================================================================
/// exp(x)
/// computes the exponential value of `x`
///
/// - Parameter x: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func exp<T>(_ x: T) -> T where
    T: TensorView, T.Element: Real
{
    var result = x.createDense()
    DeviceContext.currentQueue.exp(x: x, result: &result)
    return result
}

public extension TensorView where Element: Real {
    @inlinable @inline(__always)
    func exp(_ x: Self) -> Self { SwiftRT.exp(x) }

    @inlinable @inline(__always)
    func exp() -> Self { exp(self) }
}

//==============================================================================
/// log(x)
/// computes the log of `x`
///
/// with placement
/// - Parameter x: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func log<T>(_ x: T) -> T where
    T: TensorView, T.Element: Real
{
    var result = x.createDense()
    DeviceContext.currentQueue.log(x: x, result: &result)
    return result
}

public extension TensorView where Element: Real {
    @inlinable @inline(__always)
    func log(_ x: Self) -> Self { SwiftRT.log(x) }

    @inlinable @inline(__always)
    func log() -> Self { log(self) }
}

//==============================================================================
/// neg(x)
/// computes the negated value of `x`
///
/// with placement
/// - Parameter x: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func neg<T>(_ x: T) -> T where
    T: TensorView, T.Element: FloatingPoint
{
    var result = x.createDense()
    DeviceContext.currentQueue.neg(x: x, result: &result)
    return result
}

public extension TensorView where Element: FloatingPoint {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable @inline(__always)
    func neg() -> Self { SwiftRT.neg(self) }

    @differentiable(where Self: DifferentiableTensorView)
    @inlinable @inline(__always)
    static prefix func - (x: Self) -> Self { x.neg() }
}

@inlinable @inline(__always)
@differentiating(neg)
internal func _vjpNeg<T>(_ x: T) -> (value: T, pullback: (T) -> T) where
    T: DifferentiableTensorView
{
  (-x, { v in -v })
}

//==============================================================================
/// squared(x)
/// computes the elementwise squares of `x`
///
/// - Parameter x: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func squared<T>(_ x: T) -> T where
    T: TensorView, T.Element: Numeric
{
    var result = x.createDense()
    DeviceContext.currentQueue.squared(x: x, result: &result)
    return result
}

public extension TensorView where Element: Numeric {
    @inlinable @inline(__always)
    func squared() -> Self { SwiftRT.squared(self) }
}

//==============================================================================
/// pow(x)
/// computes elementwise `x` to the power of `y`
///
/// - Parameter x: value tensor
/// - Parameter y: power tensor
/// - Returns: result
@inlinable @inline(__always)
public func pow<T>(_ x: T, _ y: T) -> T where
    T: TensorView, T.Element: Real
{
    assert(x.extents == y.extents, _messageTensorExtentsMismatch)
    var result = x.createDense()
    DeviceContext.currentQueue.squared(x: x, result: &result)
    return result
}

public extension TensorView where Element: Real {
    @inlinable
    static func **(_ x: Self, _ y: Self) -> Self { SwiftRT.pow(x, y) }

    @inlinable
    static func **(_ x: Self, _ y: Element) -> Self {
        y == 2 ? x.squared() : x ** x.create(repeating: y)
    }

    @inlinable
    static func **(_ x: Element, _ y: Self) -> Self {
        y.create(repeating: x) ** y
    }
}

//==============================================================================
/// sqrt(x)
/// computes the square root of `x`
///
/// with placement
/// - Parameter x: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func sqrt<T>(_ x: T) -> T where
    T: TensorView, T.Element: Real
{
    var result = x.createDense()
    DeviceContext.currentQueue.sqrt(x: x, result: &result)
    return result
}

public extension TensorView where Element: Real {
    @inlinable @inline(__always)
    func sqrt() -> Self { SwiftRT.sqrt(self) }
}

//==============================================================================
/// Derivative registration
extension TensorView where Self: DifferentiableTensorView {
    @differentiating(squared)
    @inlinable @inline(__always)
    func vjpSquared() -> (value: Self, pullback: (Self) -> (Self)) {
        return (squared(), { v in v * (self + self) })
    }
}
