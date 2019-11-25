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
/// max
/// Computes the element-wise maximum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable @inline(__always)
public func max<T>(_ lhs: T, _ rhs: T) -> T where
    T: TensorView, T.Element: Comparable
{
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createDense()
    DeviceContext.currentQueue.max(lhs: lhs, rhs: rhs, result: &result)
    return result
}

@inlinable @inline(__always)
public func max<T>(_ lhs: T, _ rhs: T.Element) -> T where
    T: TensorView, T.Element: Comparable
{
    max(lhs, lhs.create(repeating: rhs))
}

@inlinable @inline(__always)
public func max<T>(_ lhs: T.Element, _ rhs: T) -> T where
    T: TensorView, T.Element: Comparable
{
    max(rhs.create(repeating: lhs), rhs)
}

////--------------------------------------
//// derivative functions
//@differentiating(max)
//@inlinable @inline(__always)
//func _vjpMax<T>(_ lhs: T, _ rhs: T)
//    -> (value: T, pullback: (T) -> (T, T)) where
//    T: DifferentiableTensorView & Comparable
//{
//    let value = SwiftRT.max(lhs, rhs)
//    // FIXME: Implement pullback.
//    return (value, { v in fatalError() })
//}

//@differentiating(max)
//@inlinable @inline(__always)
//func vjpMax<T>(_ lhs: T, _ rhs: T.Element) -> (
//    value: T, pullback: (T) -> (T, T.Element)
//    ) where
//    T: DifferentiableTensorView & Comparable
//{
//    let value = max(lhs, rhs)
//    // FIXME: Implement pullback.
//    return (value, { v in fatalError() })
//}
//
//@differentiating(max)
//@inlinable @inline(__always)
//func _vjpMax<T>(_ lhs: T.Element, _ rhs: T) -> (
//    value: T, pullback: (T) -> (T.Element, T)) where
//    T: DifferentiableTensorView & Comparable
//{
//    let value = max(lhs, rhs)
//    // FIXME: Implement pullback.
//    return (value, { v in fatalError() })
//}
//

//==============================================================================
/// min
/// Computes the element-wise minimum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable @inline(__always)
public func min<T>(_ lhs: T, _ rhs: T) -> T where
    T: TensorView, T.Element: Comparable
{
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createDense()
    DeviceContext.currentQueue.min(lhs: lhs, rhs: rhs, result: &result)
    return result
}

@inlinable @inline(__always)
public func min<T>(_ lhs: T, _ rhs: T.Element) -> T
    where T: TensorView, T.Element: Comparable
{
    min(lhs, lhs.create(repeating: rhs))
}

@inlinable @inline(__always)
public func min<T>(_ lhs: T.Element, _ rhs: T) -> T
    where T: TensorView, T.Element: Comparable
{
    min(rhs.create(repeating: lhs), rhs)
}

////--------------------------------------
//// derivative functions
//@differentiating(min)
//@inlinable @inline(__always)
//func vjpMinimum<T>(_ lhs: T, _ rhs: T) -> (
//    value: T, pullback: (T) -> (T, T)
//    ) where
//    T: DifferentiableTensorView
//{
//    let value = min(lhs, rhs)
//    // FIXME: Implement pullback.
//    return (value, { v in fatalError() })
//}
//
//@differentiating(min)
//@inlinable @inline(__always)
//func vjpMinimum<T>(_ lhs: T, _ rhs: T.Element) -> (
//    value: T, pullback: (T) -> (T, T.Element)
//    ) where
//    T: DifferentiableTensorView
//{
//    let value = min(lhs, rhs)
//    // FIXME: Implement pullback.
//    return (value, { v in fatalError() })
//}
//
//@differentiating(min)
//@inlinable @inline(__always)
//func vjpMinimum<T>(_ lhs: T.Element, _ rhs: T) -> (
//    value: T, pullback: (T) -> (T.Element, T)
//    ) where
//    T: DifferentiableTensorView
//{
//    let value = min(lhs, rhs)
//    // FIXME: Implement pullback.
//    return (value, { v in fatalError() })
//}

//==============================================================================
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

//==============================================================================
/// elementsAlmostEqual
/// Performs element-wise equality comparison within the tolerance range
/// and returns a tensor of Bool values
public func elementsAlmostEqual<T>(_ lhs: T, _ rhs: T,
                                   tolerance: T.Element) -> T.BoolView where
    T: TensorView, T.Element: SignedNumeric & Comparable
{
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createBoolTensor()
    DeviceContext.currentQueue.elementsAlmostEqual(lhs: lhs, rhs: rhs,
                                                   tolerance: tolerance,
                                                   result: &result)
    return result
}

public extension TensorView where Element: SignedNumeric & Comparable {
    func elementsAlmostEqual(_ other: Self, tolerance: Element) -> BoolView {
        SwiftRT.elementsAlmostEqual(self, other, tolerance: tolerance)
    }
}

//==============================================================================
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
