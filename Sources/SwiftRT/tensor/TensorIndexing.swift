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

public extension TensorView {
    @inlinable @inline(__always)
    func resolve(index array: Shape.Array) -> Shape.Array {
        var result = array
        for i in 0..<rank where array[i] < 0 { result[i] += extents[i] }
        return result
    }
    
    @inlinable @inline(__always)
    func resolve(extents array: Shape.Array) -> Shape.Array {
        var result = array
        for i in 0..<rank where array[i] < 1 { result[i] += extents[i] }
        return result
    }

    //--------------------------------------------------------------------------
    /// makeStepped(view:parent:steps:
    /// computes the extents and strides for creating a stepped subview
    /// - Parameter view: the extents of the desired view in parent coordinates
    /// - Parameter steps: the step interval along each dimension
    /// - Returns: the extents and strides to be used to create a subview
    func makeStepped(view extents: Shape.Array, steps: Shape.Array) ->
        (extents: Shape.Array, strides: Shape.Array)
    {
        var subExtents = extents
        zip(extents, steps).enumerated().forEach {
            subExtents[$0] = $1.0 / $1.1 + ($1.0 % $1.1 == 0 ? 0 : 1)
        }

        var subStrides = strides
        zip(strides, steps).enumerated().forEach {
            subStrides[$0] = $1.0 * $1.1
        }
        return (subExtents, subStrides)
    }

    //--------------------------------------------------------------------------
    @inlinable @inline(__always)
//    @differentiable(where Self: DifferentiableTensorView)
    subscript(index: Shape.Tuple, extents: Shape.Tuple) -> Self {
        get {
            self[resolve(index: Shape.Array(index)),
                 resolve(extents: Shape.Array(extents))]
        }
        
        set {
            self[resolve(index: Shape.Array(index)),
                 resolve(extents: Shape.Array(extents))] = newValue
        }
    }

    //--------------------------------------------------------------------------
    @inlinable @inline(__always)
    @differentiable(vjp: _vjpSubscript where Self: DifferentiableTensorView)
    subscript(index: Shape.Array, extents: Shape.Array) -> Self {
        // views will have the same isShared state as the parent
        get {
            createView(at: index, extents: extents, strides: strides,
                       isReference: isShared)
        }
        set {
            var view = createView(at: index, extents: extents, strides: strides,
                                  isReference: isShared)
            copy(from: newValue, to: &view)
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable @inline(__always)
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript(index: Shape.Tuple, extents: Shape.Tuple, steps: Shape.Tuple) -> Self {
        get {
            self[resolve(index: Shape.Array(index)),
                 resolve(extents: Shape.Array(extents)),
                 resolve(extents: Shape.Array(steps))]
        }
        
        set {
            self[resolve(index: Shape.Array(index)),
                 resolve(extents: Shape.Array(extents)),
                 resolve(extents: Shape.Array(steps))] = newValue
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable @inline(__always)
    @differentiable(vjp: _vjpSubscript where Self: DifferentiableTensorView)
    subscript(index: Shape.Array, extents: Shape.Array, steps: Shape.Array) -> Self {
        // views will have the same isShared state as the parent
        get {
            let (extents, strides) = makeStepped(view: extents, steps: steps)
            return createView(at: index, extents: extents, strides: strides,
                              isReference: isShared)
        }
        set {
            let (extents, strides) = makeStepped(view: extents, steps: steps)
            var view = createView(at: index, extents: extents, strides: strides,
                                  isReference: isShared)
            copy(from: newValue, to: &view)
        }
    }
}

//==============================================================================
/// Derivative registration
extension TensorView where Self: DifferentiableTensorView {
    // https://github.com/apple/swift/blob/37b507b31c77ef969151f385cd1902dd44fb3b7f/stdlib/public/core/Array.swift#L2091
    @inlinable @inline(__always)
    func _vjpSubscript(index: Shape.Array, extents: Shape.Array)
        -> (value: Self, pullback: (Self) -> Self)
    {
        return (self[index, extents], { v in
            var result = self.filled(with: 0)
            result[index, extents] = v
            return result
        })
    }
    
    @inlinable @inline(__always)
    func _vjpSubscript(index: Shape.Array, extents: Shape.Array, steps: Shape.Array)
        -> (value: Self, pullback: (Self) -> Self)
    {
        return (self[index, extents, steps], { v in
            var result = self.filled(with: 0)
            result[index, extents, steps] = v
            return result
        })
    }
}

//==============================================================================
public protocol TensorIndexing: Strideable {
    associatedtype Position
    /// sequential logical view element index
    var viewIndex: Int { get }
    /// linear data buffer element index
    var dataIndex: Int { get }

    /// initializer for starting at any position
    init<T>(view: T, at position: Position) where T: TensorView
    /// initializer specifically for the endIndex
    init<T>(endOf view: T) where T: TensorView
    
    /// highest frequency function to move the index
    /// use advanced(by n: for jumps or negative movement
    func increment() -> Self
}

public extension TensorIndexing {
    // Equatable
    @inlinable @inline(__always)
    static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs.viewIndex == rhs.viewIndex
    }
    
    // Comparable
    @inlinable @inline(__always)
    static func < (lhs: Self, rhs: Self) -> Bool {
        return lhs.viewIndex < rhs.viewIndex
    }
    
    @inlinable @inline(__always)
    func distance(to other: Self) -> Int {
        return other.viewIndex - viewIndex
    }
}

//==============================================================================
/// TensorValueCollection
public struct TensorValueCollection<View>: RandomAccessCollection
    where View: TensorView
{
    // properties
    public let view: View
    public let buffer: UnsafeBufferPointer<View.Element>
    public let startIndex: View.Index
    public let endIndex: View.Index
    public let count: Int

    public init(view: View, buffer: UnsafeBufferPointer<View.Element>) {
        self.view = view
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.count
    }

    public init(view: View) {
        self.view = view
        self.buffer = UnsafeBufferPointer<View.Element>(start: nil, count: 0)
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.count
    }
    
    //--------------------------------------------------------------------------
    // Collection
    @inlinable @inline(__always)
    public func index(before i: View.Index) -> View.Index {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    public func index(after i: View.Index) -> View.Index {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    public subscript(index: View.Index) -> View.Element {
        return buffer[index.dataIndex]
    }
}

//==============================================================================
/// TensorMutableValueCollection
public struct TensorMutableValueCollection<View>: RandomAccessCollection,
    MutableCollection where View: TensorView
{
    // properties
    public let buffer: UnsafeMutableBufferPointer<View.Element>
    public let startIndex: View.Index
    public let endIndex: View.Index
    public let count: Int
    
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<View.Element>) {
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.count
    }
    
    public init(view: inout View) {
        self.buffer = UnsafeMutableBufferPointer<View.Element>(start: nil,
                                                               count: 0)
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.count
    }
    
    //--------------------------------------------------------------------------
    // Collection
    @inlinable @inline(__always)
    public func index(before i: View.Index) -> View.Index {
        return i.advanced(by: -1)
    }
    
    @inlinable @inline(__always)
    public func index(after i: View.Index) -> View.Index {
        return i.increment()
    }
    
    @inlinable @inline(__always)
    public subscript(index: View.Index) -> View.Element {
        get {
            return buffer[index.dataIndex]
        }
        set {
            buffer[index.dataIndex] = newValue
        }
    }
}
