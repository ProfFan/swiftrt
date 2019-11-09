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
/// Computes the element-wise maximum of two tensors.
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the extents are smaller than `lhs`
///   then broadcasting will be performed via repeated indexing.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpAdd(lhs:rhs:) where Element : TensorFlowFloatingPoint)
public func maximum<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element: Comparable
{
    DeviceContext.currentQueue.maximum(lhs: lhs, rhs: rhs, result: &result)
}

/// returns new view
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the extents are smaller than
///   `lhs` then broadcasting is performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func maximum<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: Comparable
{
    var result = lhs.createDense()
    maximum(lhs: lhs, rhs: rhs, result: &result)
    return result
}

/// returns new view
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand scalar
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func maximum<T>(_ lhs: T, _ rhs: T.Element) -> T
    where T: TensorView, T.Element: Comparable
{
    var result = lhs.createDense()
    maximum(lhs: lhs, rhs: lhs.create(repeating: rhs), result: &result)
    return result
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
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceQueue])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func maximum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    {
        queue(#function, { (lhs.values(), rhs.values()) }, &result) {
            zip($0.0, $0.1).map(into: &$1) { $0 >= $1 ? $0 : $1 }
        }
    }
}
#endif

//==============================================================================
/// minimum
/// Computes the element-wise maximum of two tensors.
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the extents are smaller than `lhs`
///   then broadcasting will be performed via repeated indexing.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpAdd(lhs:rhs:) where Element : TensorFlowFloatingPoint)
public func minimum<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element: Comparable
{
    DeviceContext.currentQueue.minimum(lhs: lhs, rhs: rhs, result: &result)
}

/// returns new view
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand scalar
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func minimum<T>(_ lhs: T, _ rhs: T.Element) -> T
    where T: TensorView, T.Element: Comparable
{
    var result = lhs.createDense()
    minimum(lhs: lhs, rhs: lhs.create(repeating: rhs), result: &result)
    return result
}

/// returns new view
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the extents are smaller than
///   `lhs` then broadcasting is performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func minimum<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: Comparable
{
    var result = lhs.createDense()
    minimum(lhs: lhs, rhs: rhs, result: &result)
    return result
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
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceQueue])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func minimum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    {
        queue(#function, { (lhs.values(), rhs.values()) }, &result) {
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
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func exp<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: AnyFloatingPoint
{
    DeviceContext.currentQueue.exp(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func exp<T>(_ x: T) -> T
    where T: TensorView, T.Element: AnyFloatingPoint
{
    var result = x.createDense()
    exp(x, result: &result)
    return result
}

public extension TensorView where Element: AnyFloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpAbs(_:) where T: TensorFlowFloatingPoint)
    func exp() -> Self {
        var result = createDense()
        SwiftRT.exp(self, result: &result)
        return result
    }
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
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceQueue])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// exp
    func exp<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    {
        queue(#function, { x.values() }, &result) {
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
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func log<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: AnyFloatingPoint
{
    DeviceContext.currentQueue.log(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func log<T>(_ x: T) -> T
    where T: TensorView, T.Element: AnyFloatingPoint
{
    var result = x.createDense()
    log(x, result: &result)
    return result
}

public extension TensorView where Element: AnyFloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpAbs(_:) where T: TensorFlowFloatingPoint)
    func log() -> Self {
        var result = createDense()
        SwiftRT.log(self, result: &result)
        return result
    }
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
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceQueue])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// log
    func log<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    {
        queue(#function, { x.values() }, &result) {
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
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func neg<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    DeviceContext.currentQueue.neg(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func neg<T>(_ x: T) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    var result = x.createDense()
    neg(x, result: &result)
    return result
}

public extension TensorView where Element: FloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpAbs(_:) where T: TensorFlowFloatingPoint)
    func neg() -> Self {
        var result = createDense()
        SwiftRT.neg(self, result: &result)
        return result
    }
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
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceQueue])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// neg
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Element: SignedNumeric
    {
        queue(#function, { x.values() }, &result) {
            $0.map(into: &$1) { -$0 }
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
/// equal
/// Computes `lhs == rhs` element-wise and returns a `TensorView` of Boolean
/// values.
public func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Element: Equatable
{
    assert(lhs.shape == rhs.shape, "shapes must match")
    DeviceContext.currentQueue.equal(lhs: lhs, rhs: rhs, result: &result)
}

public extension TensorView where Element: Equatable & AnyScalar {
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: a new tensor containing the result
    @inlinable
    static func .== (_ lhs: Self, _ rhs: Self) -> BoolView {
        var result = lhs.createBoolTensor()
        equal(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: `true` if the tensors are equal
    @inlinable
    static func == (lhs: Self, rhs: Self) -> Bool {
        // the shapes must match or they are not equal
        guard lhs.shape == rhs.shape else { return false }
        
        // if lhs is an alias for rhs, then they match
        if lhs.tensorArray === rhs.tensorArray &&
            lhs.viewOffset == rhs.viewOffset { return true }
        
        // compare elements
        return (lhs .== rhs).all().element
    }

    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: `true` if the tensors are not equal
    @inlinable
    static func != (lhs: Self, rhs: Self) -> Bool {
        return !(lhs == rhs)
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
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceQueue])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    {
        queue(#function, { (lhs.values(), rhs.values()) }, &result) {
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
public func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Element: Equatable
{
    assert(lhs.shape == rhs.shape, "shapes must match")
    DeviceContext.currentQueue.notEqual(lhs: lhs, rhs: rhs, result: &result)
}

public extension TensorView where Element: Equatable {
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: a new tensor containing the result
    @inlinable
    static func .!= (_ lhs: Self, _ rhs: Self) -> BoolView {
        var result = lhs.createBoolTensor()
        notEqual(lhs: lhs, rhs: rhs, result: &result)
        return result
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
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceQueue])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    /// equal
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    {
        queue(#function, { (lhs.values(), rhs.values()) }, &result) {
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
/// with placement
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
public func squared<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: Numeric
{
    DeviceContext.currentQueue.squared(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
public func squared<T>(_ x: T) -> T
    where T: TensorView, T.Element: Numeric
{
    var result = x.createDense()
    squared(x, result: &result)
    return result
}

public extension TensorView where Element: Numeric {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    func squared() -> Self {
        var result = createDense()
        SwiftRT.squared(self, result: &result)
        return result
    }
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
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceQueue])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func squared<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    {
        queue(#function, { x.values() }, &result) {
            $0.map(into: &$1) { $0 * $0 }
        }
    }
}
#endif

