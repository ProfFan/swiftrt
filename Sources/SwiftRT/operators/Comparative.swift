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
        zip(lhs, rhs).map(into: &result) { $0 >= $1 ? $0 : $1 }
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

public extension TensorView where Element: Equatable {
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
        zip(lhs, rhs).map(into: &result, ==)
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
            zip($0.0, $0.1).map(into: &$1, ==)
        }
    }
}
#endif

//elementsAlmostEqual(expected, tolerance: tolerance)


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

    // ambiguous with AdditiveArithmetic
//    @inlinable
//    static func != (lhs: Self, rhs: Self) -> Bool {
//        // the extents must not match or they are not equal
//        guard lhs.extents != rhs.extents else { return true }
//
//        // if lhs is an alias for rhs, then they match
//        if (lhs.tensorArray === rhs.tensorArray &&
//            lhs.viewOffset == rhs.viewOffset) { return false }
//
//        // compare elements
//        return (lhs .!= rhs).any().element
//    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    /// notEqual
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    {
        zip(lhs, rhs).map(into: &result, !=)
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
            zip($0.0, $0.1).map(into: &$1, !=)
        }
    }
}
#endif

//==============================================================================
/// Derivative registration

@differentiating(maximum)
@inlinable @inline(__always)
func vjpMaximum<T>(_ lhs: T, _ rhs: T) -> (
    value: T, pullback: (T) -> (T, T)
) where
    T: DifferentiableTensorView
{
    let value = maximum(lhs, rhs)
    // FIXME: Implement pullback.
    return (value, { v in fatalError() })
}

@differentiating(maximum)
@inlinable @inline(__always)
func vjpMaximum<T>(_ lhs: T, _ rhs: T.Element) -> (
    value: T, pullback: (T) -> (T, T.Element)
) where
    T: DifferentiableTensorView
{
    let value = maximum(lhs, rhs)
    // FIXME: Implement pullback.
    return (value, { v in fatalError() })
}

@differentiating(maximum)
@inlinable @inline(__always)
func vjpMaximum<T>(_ lhs: T.Element, _ rhs: T) -> (
    value: T, pullback: (T) -> (T.Element, T)) where
    T: DifferentiableTensorView
{
    let value = maximum(lhs, rhs)
    // FIXME: Implement pullback.
    return (value, { v in fatalError() })
}

@differentiating(minimum)
@inlinable @inline(__always)
func vjpMinimum<T>(_ lhs: T, _ rhs: T) -> (
    value: T, pullback: (T) -> (T, T)
) where
    T: DifferentiableTensorView
{
    let value = minimum(lhs, rhs)
    // FIXME: Implement pullback.
    return (value, { v in fatalError() })
}

@differentiating(minimum)
@inlinable @inline(__always)
func vjpMinimum<T>(_ lhs: T, _ rhs: T.Element) -> (
    value: T, pullback: (T) -> (T, T.Element)
) where
    T: DifferentiableTensorView
{
    let value = minimum(lhs, rhs)
    // FIXME: Implement pullback.
    return (value, { v in fatalError() })
}

@differentiating(minimum)
@inlinable @inline(__always)
func vjpMinimum<T>(_ lhs: T.Element, _ rhs: T) -> (
    value: T, pullback: (T) -> (T.Element, T)
) where
    T: DifferentiableTensorView
{
    let value = minimum(lhs, rhs)
    // FIXME: Implement pullback.
    return (value, { v in fatalError() })
}
