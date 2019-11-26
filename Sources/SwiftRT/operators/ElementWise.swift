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
public func exp<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    var result = x.createDense()
    DeviceContext.currentQueue.exp(x: x, result: &result)
    return result
}

public extension TensorView where Element: Real {
    @inlinable @inline(__always)
    func exp() -> Self { SwiftRT.exp(self) }
}

//--------------------------------------
// derivative functions
@differentiating(exp)
@inlinable @inline(__always)
internal func _vjpExp<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    let value = exp(x)
    return (value, { v in value * v } )
}

extension TensorView where Self: DifferentiableTensorView, Element: Real {
    @differentiating(exp)
    @inlinable @inline(__always)
    func _vjpExp() -> (value: Self, pullback: (Self) -> (Self)) {
        SwiftRT._vjpExp(self)
    }
}

//==============================================================================
/// log(x)
/// computes the log of `x`
///
/// - Parameter x: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func log<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    var result = x.createDense()
    DeviceContext.currentQueue.log(x: x, result: &result)
    return result
}

public extension TensorView where Element: Real {
    @inlinable @inline(__always)
    func log() -> Self { SwiftRT.log(self) }
}

//--------------------------------------
// derivative functions
@differentiating(log)
@inlinable @inline(__always)
internal func _vjpLog<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    (log(x), { v in v / x })
}

extension TensorView where Self: DifferentiableTensorView, Element: Real {
    @differentiating(log)
    @inlinable @inline(__always)
    func _vjpLog() -> (value: Self, pullback: (Self) -> (Self)) {
        SwiftRT._vjpLog(self)
    }
}

//==============================================================================
/// neg(x)
/// computes the negated value of `x`
///
/// with placement
/// - Parameter x: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func neg<T>(_ x: T) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    var result = x.createDense()
    DeviceContext.currentQueue.neg(x: x, result: &result)
    return result
}

public extension TensorView where Element: FloatingPoint {
    @inlinable @inline(__always)
    func neg() -> Self { SwiftRT.neg(self) }

    @inlinable @inline(__always)
    static prefix func - (x: Self) -> Self { x.neg() }
}

//--------------------------------------
// derivative functions
@inlinable
@differentiating(neg)
internal func _vjpNeg<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView
{
    (-x, { v in -v })
}

extension TensorView where Self: DifferentiableTensorView {
    @differentiating(neg)
    @inlinable @inline(__always)
    func _vjpNeg() -> (value: Self, pullback: (Self) -> (Self)) {
        SwiftRT._vjpNeg(self)
    }
    
    @differentiating(-)
    @inlinable @inline(__always)
    static func _vjpNeg(x: Self) -> (value: Self, pullback: (Self) -> (Self))
    {
        SwiftRT._vjpNeg(x)
    }
}

//==============================================================================
/// squared(x)
/// computes the elementwise squares of `x`
///
/// - Parameter x: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func squared<T>(_ x: T) -> T
    where T: TensorView, T.Element: Numeric
{
    var result = x.createDense()
    DeviceContext.currentQueue.squared(x: x, result: &result)
    return result
}

public extension TensorView where Element: Numeric {
    @inlinable @inline(__always)
    func squared() -> Self { SwiftRT.squared(self) }
}

//--------------------------------------
// derivative functions
@inlinable
@differentiating(squared)
internal func _vjpSquared<T>(_ x: T) -> (value: T, pullback: (T) -> (T))
    where T: DifferentiableTensorView
{
    (squared(x), { v in v * (x + x) })
}

extension TensorView where Self: DifferentiableTensorView {
    @differentiating(squared)
    @inlinable @inline(__always)
    func _vjpSquared() -> (value: Self, pullback: (Self) -> (Self)) {
        SwiftRT._vjpSquared(self)
    }
}

//==============================================================================
/// pow(x)
/// computes elementwise `x` to the power of `y`
///
/// - Parameter x: value tensor
/// - Parameter y: power tensor
/// - Returns: result
@inlinable @inline(__always)
public func pow<T>(_ x: T, _ y: T) -> T
    where T: TensorView, T.Element: Real
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
        y == 2 ? x.squared() : x ** Self(repeating: y, like: x)
    }

    @inlinable
    static func **(_ x: Element, _ y: Self) -> Self {
        Self(repeating: x, like: y) ** y
    }
}

//--------------------------------------
// derivative functions
@inlinable
@differentiating(pow)
internal func _vjpPow<T>(_ x: T, _ y: T) -> (value: T, pullback: (T) -> (T, T))
    where T: DifferentiableTensorView, T.Element: Real
{
    let value = pow(x, y)
    return (value, { v in
        let safeX = x.replacing(with: 1, where: x .<= 0)
        let lhsGrad = v * y * pow(x, y - 1)
        let rhsGrad = value * v * log(safeX)
        return (T(repeating: lhsGrad.sum().element, like: x),
                T(repeating: rhsGrad.sum().element, like: y))
    })
}

//==============================================================================
/// sqrt(x)
/// computes the square root of `x`
///
/// with placement
/// - Parameter x: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func sqrt<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    var result = x.createDense()
    DeviceContext.currentQueue.sqrt(x: x, result: &result)
    return result
}

public extension TensorView where Element: Real {
    @inlinable @inline(__always)
    func sqrt() -> Self { SwiftRT.sqrt(self) }
}
