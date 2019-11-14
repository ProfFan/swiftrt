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
/// maximum
/// Computes the element-wise maximum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable @inline(__always)
public func maximum<T>(_ lhs: T, _ rhs: T) -> T where
    T: TensorView, T.Element: Comparable
{
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createDense()
    DeviceContext.currentQueue.maximum(lhs: lhs, rhs: rhs, result: &result)
    return result
}

@inlinable @inline(__always)
public func maximum<T>(_ lhs: T, _ rhs: T.Element) -> T where
    T: TensorView, T.Element: Comparable
{
    maximum(lhs, lhs.create(repeating: rhs))
}

@inlinable @inline(__always)
public func maximum<T>(_ lhs: T.Element, _ rhs: T) -> T where
    T: TensorView, T.Element: Comparable
{
    maximum(rhs.create(repeating: lhs), rhs)
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    func maximum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    {
        zip(lhs, rhs).map(into: &result) { $0 <= $1 ? $0 : $1 }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func maximum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    {
        queue(#function, {
            (lhs.elements(using: self),
             rhs.elements(using: self))
        }, &result) {
            zip($0.0, $0.1).map(into: &$1) { $0 >= $1 ? $0 : $1 }
        }
    }
}
#endif

//==============================================================================
/// minimum
/// Computes the element-wise minimum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable @inline(__always)
public func minimum<T>(_ lhs: T, _ rhs: T) -> T where
    T: TensorView, T.Element: Comparable
{
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createDense()
    DeviceContext.currentQueue.minimum(lhs: lhs, rhs: rhs, result: &result)
    return result
}

@inlinable @inline(__always)
public func minimum<T>(_ lhs: T, _ rhs: T.Element) -> T
    where T: TensorView, T.Element: Comparable
{
    minimum(lhs, lhs.create(repeating: rhs))
}

@inlinable @inline(__always)
public func minimum<T>(_ lhs: T.Element, _ rhs: T) -> T
    where T: TensorView, T.Element: Comparable
{
    minimum(rhs.create(repeating: lhs), rhs)
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    func minimum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    {
        zip(lhs, rhs).map(into: &result) { $0 <= $1 ? $0 : $1 }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func minimum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    {
        queue(#function, {
            (lhs.elements(using: self),
             rhs.elements(using: self))
        }, &result) {
            zip($0.0, $0.1).map(into: &$1) { $0 <= $1 ? $0 : $1 }
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
/// exp(x)
/// computes the exponential value of `x`
///
/// with placement
/// - Parameter x: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func exp<T>(_ x: T) -> T where
    T: TensorView, T.Element: AnyFloatingPoint
{
    var result = x.createDense()
    DeviceContext.currentQueue.exp(x: x, result: &result)
    return result
}

public extension TensorView where Element: AnyFloatingPoint {
    @inlinable @inline(__always)
    func exp() -> Self { SwiftRT.exp(self) }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    /// exp
    func exp<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    {
        x.map(into: &result) { T.Element(any: Foundation.exp($0.asFloat)) }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// exp
    func exp<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    {
        queue(#function, { x.elements(using: self) }, &result) {
            $0.map(into: &$1) {
                T.Element(any: Foundation.exp($0.asFloat))
            }
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
/// log(x)
/// computes the log of `x`
///
/// with placement
/// - Parameter x: value tensor
/// - Returns: result
@inlinable @inline(__always)
public func log<T>(_ x: T) -> T where
    T: TensorView, T.Element: AnyFloatingPoint
{
    var result = x.createDense()
    DeviceContext.currentQueue.log(x: x, result: &result)
    return result
}

public extension TensorView where Element: AnyFloatingPoint {
    @inlinable @inline(__always)
    func log() -> Self { SwiftRT.log(self) }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    /// log
    func log<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    {
        x.map(into: &result) { T.Element(any: Foundation.log($0.asFloat)) }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// log
    func log<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    {
        queue(#function, { x.elements(using: self) }, &result) {
            $0.map(into: &$1) {
                T.Element(any: Foundation.log($0.asFloat))
            }
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
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
    @inlinable @inline(__always)
    func neg() -> Self { SwiftRT.neg(self) }

    @inlinable @inline(__always)
    static prefix func - (x: Self) -> Self { x.neg() }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    /// neg
    /// returns the element-wise negation of `x`
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Element: SignedNumeric
    {
        x.map(into: &result) { -$0 }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// neg
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Element: SignedNumeric
    {
        queue(#function, { x.elements(using: self) }, &result) {
            $0.map(into: &$1) { -$0 }
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
/// equal
/// Performs element-wise equality comparison and returns a
/// tensor of Bool values
public func equal<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
    T: TensorView, T.Element: Equatable
{
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createBoolTensor()
    DeviceContext.currentQueue.equal(lhs: lhs, rhs: rhs, result: &result)
    return result
}

public extension TensorView where Element: Equatable & AnyScalar {
    @inlinable
    static func .== (_ lhs: Self, _ rhs: Self) -> BoolView { equal(lhs, rhs) }
    
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: `true` if the tensors are equal
    @inlinable
    static func == (lhs: Self, rhs: Self) -> Bool {
        // the extents must match or they are not equal
        guard lhs.extents == rhs.extents else { return false }
        
        // if lhs is an alias for rhs, then they match
        if lhs.tensorArray === rhs.tensorArray &&
            lhs.viewOffset == rhs.viewOffset { return true }
        
        // compare elements
        return (lhs .== rhs).all().element
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    {
        zip(lhs, rhs).map(into: &result) { $0 == $1 }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    {
        queue(#function, {
            (lhs.elements(using: self),
             rhs.elements(using: self))
        }, &result) {
            zip($0.0, $0.1).map(into: &$1) { $0 == $1 }
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
/// notEqual
/// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
/// values.
public func notEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
    T: TensorView, T.Element: Equatable
{
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createBoolTensor()
    DeviceContext.currentQueue.notEqual(lhs: lhs, rhs: rhs, result: &result)
    return result
}

public extension TensorView where Element: Equatable {
    @inlinable
    static func .!=(_ lhs: Self, _ rhs: Self) -> BoolView { notEqual(lhs, rhs) }

    @inlinable
    static func != (lhs: Self, rhs: Self) -> Bool {
        // the extents must not match or they are not equal
        guard lhs.extents != rhs.extents else { return true }
        
        // if lhs is an alias for rhs, then they match
        if (lhs.tensorArray === rhs.tensorArray &&
            lhs.viewOffset == rhs.viewOffset) { return false }
        
        // compare elements
        return (lhs .!= rhs).any().element
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    /// notEqual
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    {
        zip(lhs, rhs).map(into: &result) { $0 != $1 }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// equal
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    {
        queue(#function, {
            (lhs.elements(using: self),
             rhs.elements(using: self))
        }, &result) {
            zip($0.0, $0.1).map(into: &$1) { $0 != $1 }
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
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

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    func squared<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    {
        x.map(into: &result) { $0 * $0 }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func squared<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    {
        queue(#function, { x.elements(using: self) }, &result) {
            $0.map(into: &$1) { $0 * $0 }
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
/// squared(x)
/// computes elementwise `x` to the power of `y`
///
/// - Parameter x: value tensor
/// - Parameter y: power tensor
/// - Returns: result
@inlinable @inline(__always)
public func pow<T>(_ x: T, _ y: T) -> T where
    T: TensorView, T.Element: AnyNumeric
{
    assert(x.extents == y.extents, _messageTensorExtentsMismatch)
    var result = x.createDense()
    DeviceContext.currentQueue.squared(x: x, result: &result)
    return result
}

public extension TensorView where Element: AnyNumeric {
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

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
public extension DeviceQueue {
    func pow<T>(x: T, y: T, result: inout T) where
        T: TensorView, T.Element: AnyNumeric
    {
        zip(x, y).map(into: &result) {
            T.Element(any: Foundation.pow($0.asDouble, $1.asDouble))
        }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func pow<T>(x: T, y: T, result: inout T) where
        T: TensorView, T.Element: AnyNumeric
    {
        queue(#function, {
            (x.elements(using: self),
             y.elements(using: self))
        }, &result) {
            zip($0.0, $0.1).map(into: &$1) {
                T.Element(any: Foundation.pow($0.asDouble, $1.asDouble))
            }
        }
    }
}
#endif
