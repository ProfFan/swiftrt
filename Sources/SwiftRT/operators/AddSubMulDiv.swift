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
import Real

//==============================================================================
/// Elementwise add tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable @inline(__always)
public func add<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: AdditiveArithmetic
{
    let (lhs, rhs) = implicitlyMatchExtents(lhs, rhs)
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createDense()
    DeviceContext.currentQueue.add(lhs: lhs, rhs: rhs, result: &result)
    return result
}

public extension TensorView where Element: AdditiveArithmetic {
    @inlinable @inline(__always)
    static func + (lhs: Self, rhs: Self) -> Self { add(lhs, rhs) }

    @inlinable @inline(__always)
    static func += (lhs: inout Self, rhs: Element) { lhs = lhs + rhs }
    
    @inlinable @inline(__always)
    static func +(lhs: Self, rhs: Element) -> Self {
        lhs + Self(repeating: rhs, like: lhs)
    }

    @inlinable @inline(__always)
    static func +(lhs: Element, rhs: Self) -> Self {
        Self(repeating: lhs, like: rhs) + rhs
    }
}

//--------------------------------------
// derivative functions
@derivative(of: add)
@inlinable @inline(__always)
public func _vjpAdd<T>(lhs: T, rhs: T) -> (value: T, pullback: (T) -> (T, T))
    where T: DifferentiableTensorView
{
    return (lhs + rhs, { v in (v, v) })
}

public extension TensorView where Self: DifferentiableTensorView {
    @derivative(of: +)
    @inlinable @inline(__always)
    static func _vjpAdd(lhs: Self, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Self, Self))
    {
        SwiftRT._vjpAdd(lhs: lhs, rhs: rhs)
    }
    
    @derivative(of: +)
    @inlinable @inline(__always)
    static func _vjpAdd(lhs: Self, rhs: Element) ->
        (value: Self, pullback: (Self) -> (Self, Element))
    {
        return (lhs + rhs, { v in (v, v.sum().element) })
    }
    
    @derivative(of: +)
    @inlinable @inline(__always)
    static func _vjpAdd(lhs: Element, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Element, Self))
    {
        return (lhs + rhs, { v in (v.sum().element, v) })
    }
}

//==============================================================================
/// Elementwise subtract tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable @inline(__always)
public func subtract<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: AdditiveArithmetic
{
    let (lhs, rhs) = implicitlyMatchExtents(lhs, rhs)
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createDense()
    DeviceContext.currentQueue.subtract(lhs: lhs, rhs: rhs, result: &result)
    return result
}

public extension TensorView where Element: AdditiveArithmetic {
    @inlinable @inline(__always)
    static func - (lhs: Self, rhs: Self) -> Self { subtract(lhs, rhs) }

    @inlinable @inline(__always)
    static func -= (lhs: inout Self, rhs: Element) { lhs = lhs - rhs }
    
    @inlinable @inline(__always)
    static func - (lhs: Self, rhs: Element) -> Self {
        lhs - Self(repeating: rhs, like: lhs)
    }

    @inlinable @inline(__always)
    static func - (lhs: Element, rhs: Self) -> Self {
        Self(repeating: lhs, like: rhs) - rhs
    }
}

//--------------------------------------
// derivative functions
public extension TensorView
    where Self: DifferentiableTensorView, Element: SignedNumeric
{
    @derivative(of: -)
    @inlinable @inline(__always)
    static func vjpSubtract(lhs: Self, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Self, Self))
    {
        return (lhs - rhs, { v in (v, -v) })
    }
    
    @derivative(of: -)
    @inlinable @inline(__always)
    static func vjpSubtract(lhs: Self, rhs: Element) ->
        (value: Self, pullback: (Self) -> (Self, Element))
    {
        return (lhs - rhs, { v in (v, -v.sum().element) })
    }
}

//==============================================================================
/// Element wise multiply tensors with broadcasting

/// in place
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the size is smaller than `lhs` then
///   broadcasting will be performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func mul<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: Numeric
{
    let (lhs, rhs) = implicitlyMatchExtents(lhs, rhs)
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createDense()
    DeviceContext.currentQueue.mul(lhs: lhs, rhs: rhs, result: &result)
    return result
}

public extension TensorView where Element: Numeric {
    @inlinable @inline(__always)
    static func * (lhs: Self, rhs: Self) -> Self { mul(lhs, rhs) }
    
    @inlinable @inline(__always)
    static func *= (lhs: inout Self, rhs: Element) { lhs = lhs * rhs }

    @inlinable @inline(__always)
    static func *= (lhs: inout Self, rhs: Self) { lhs = lhs * rhs }
    
    @inlinable @inline(__always)
    static func * (lhs: Self, rhs: Element) -> Self {
        lhs * Self(repeating: rhs, like: lhs)
    }

    @inlinable @inline(__always)
    static func * (lhs: Element, rhs: Self) -> Self {
        Self(repeating: lhs, like: rhs) * rhs
    }
}

//--------------------------------------
// derivative functions
@derivative(of: mul)
@inlinable @inline(__always)
internal func _vjpMultiply<T>(_ lhs: T, _ rhs: T) ->
    (value: T, pullback: (T) -> (T, T)) where T: DifferentiableTensorView
{
    (lhs * rhs, { v in (v * rhs, v * lhs) })
}

public extension TensorView where Self: DifferentiableTensorView {
    @derivative(of: *)
    @inlinable @inline(__always)
    static func _vjpMultiply(lhs: Self, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Self, Self))
    {
        SwiftRT._vjpMultiply(lhs, rhs)
    }
    
    @derivative(of: *)
    @inlinable @inline(__always)
    static func _vjpMultiply(lhs: Self, rhs: Element) ->
        (value: Self, pullback: (Self) -> (Self, Element))
    {
        return (lhs * rhs, { v in (v * rhs, (v * lhs).sum().element) })
    }
    
    @derivative(of: *)
    @inlinable @inline(__always)
    static func _vjpMultiply(lhs: Element, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Element, Self))
    {
        return (lhs * rhs, { v in ((v * rhs).sum().element, v * lhs) })
    }
}

//==============================================================================
/// Element wise divide
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func div<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: Field
{
    let (lhs, rhs) = implicitlyMatchExtents(lhs, rhs)
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createDense()
    DeviceContext.currentQueue.div(lhs: lhs, rhs: rhs, result: &result)
    return result
}

public extension TensorView where Element: Field {
    @inlinable @inline(__always)
    static func / (lhs: Self, rhs: Self) -> Self { div(lhs, rhs) }

    @inlinable @inline(__always)
    static func /= (lhs: inout Self, rhs: Element) { lhs = lhs / rhs }

    @inlinable @inline(__always)
    static func /= (lhs: inout Self, rhs: Self) { lhs = lhs / rhs }

    @inlinable @inline(__always)
    static func / (lhs: Self, rhs: Element) -> Self {
        lhs / Self(repeating: rhs, like: lhs)
    }

    @inlinable @inline(__always)
    static func / (lhs: Element, rhs: Self) -> Self {
        Self(repeating: lhs, like: rhs) / rhs
    }
}

//--------------------------------------
// derivative functions
@derivative(of: div)
@inlinable @inline(__always)
internal func _vjpDivide<T>(_ lhs: T, _ rhs: T) ->
    (value: T, pullback: (T) -> (T, T))
    where T: DifferentiableTensorView, T.Element: SignedNumeric
{
    (lhs / rhs, { v in (v / rhs, -lhs / rhs.squared() * v) })
}

public extension TensorView
    where Self: DifferentiableTensorView, Element: SignedNumeric
{
    @derivative(of: /)
    @inlinable @inline(__always)
    static func _vjpDivide(lhs: Self, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Self, Self))
    {
        SwiftRT._vjpDivide(lhs, rhs)
    }
    
    @derivative(of: /)
    @inlinable @inline(__always)
    static func _vjpDivide(lhs: Self, rhs: Element) ->
        (value: Self, pullback: (Self) -> (Self, Element))
    {
        return (lhs / rhs, { v in
            (v / rhs, (-lhs / rhs.squared() * v).sum().element)
        })
    }
    
    @derivative(of: /)
    @inlinable @inline(__always)
    static func _vjpDivide(lhs: Element, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Element, Self))
    {
        return (lhs / rhs, { v in
            ((v / rhs).sum().element, -lhs / rhs.squared() * v)
        })
    }
}
