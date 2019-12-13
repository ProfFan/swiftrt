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
// VectorView protocol
public protocol VectorView: TensorView where Shape == Shape1 { }

extension Vector: Codable where Element: Codable {}

extension Vector: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

extension Vector: Equatable where Element: Equatable { }

extension Vector: AdditiveArithmetic where Element: Numeric {
    public static var zero: Vector<Element> {
        Vector<Element>(element: Element.zero)
    }
}

//==============================================================================
// MatrixView extensions
public extension VectorView {
    //--------------------------------------------------------------------------
    /// reserved space
    init(extents: Shape.Array, name: String? = nil) {
        self = Self.create(Shape(extents: extents), name)
    }
    
    init(extents: Shape.Tuple, name: String? = nil) {
        self.init(extents: Shape.Array(extents), name: name)
    }
    
    init(count: Int, name: String? = nil) {
        self.init(extents: (count), name: name)
    }
    
    //--------------------------------------------------------------------------
    /// from single `Element`
    init(element: Element, name: String? = nil) {
        self = Self.create([element], Shape(extents: (1)), name)
    }
    
    //--------------------------------------------------------------------------
    /// from single `AnyConvertable`
    init<T>(with element: T, name: String? = nil) where
        T: AnyConvertable, Element: AnyConvertable
    {
        self = Self.create([Element(any: element)], Shape(extents: (1)), name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    init<C>(elements: C, name: String? = nil) where
        C: Collection, C.Element == Element
    {
        self = Self.create(elements, Shape(extents: (elements.count)), name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    init<C>(with elements: C, name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self = Self.create(elements.lazy.map { Element(any: $0) },
                           Shape(extents: (elements.count)), name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(referenceTo buffer: UnsafeBufferPointer<Element>, name: String? = nil)
    {
        let shape = Shape(extents: (buffer.count))
        self = Self.create(referenceTo: buffer, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read write buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
         name: String? = nil)
    {
        let shape = Shape(extents: (buffer.count))
        self = Self.create(referenceTo: buffer, shape, name)
    }
    
    //--------------------------------------------------------------------------
    // typed views
    func createBoolTensor(with extents: Shape.Array) -> Vector<Bool> {
        Vector<Bool>(extents: extents)
    }
    
    func createIndexTensor(with extents: Shape.Array) -> Vector<IndexElement> {
        Vector<IndexElement>(extents: extents)
    }
}

//==============================================================================
// range subscripting
public extension VectorView {
    //--------------------------------------------------------------------------
    // TODO: probably move these off onto the TensorViewCollection
    var startIndex: VectorIndex { return VectorIndex(view: self, at: (0)) }
    var endIndex: VectorIndex { return VectorIndex(endOf: self) }
    
    @inlinable @inline(__always)
    subscript(r: UnboundedRange) -> Self { self }

    // TODO(TF-281): Rewrite `@differentiable` attribute as a `@differentiating`
    // attribute when `@differentiating` supports subscript declaration
    // references.
    @differentiable(vjp: _vjpSubscript where Self: DifferentiableTensorView)
    @inlinable @inline(__always)
    subscript<R>(r: R) -> Self where
        R: RangeExpression, R.Bound == Int
    {
        get {
            let r = resolve(range: r, count: extents[0])
            return view(at: (r.lowerBound), extents: (r.count))
        }
        set {
            let r = resolve(range: r, count: extents[0])
            var v = view(at: (r.lowerBound), extents: (r.count))
            copy(from: newValue, to: &v)
        }
    }
    
    @inlinable @inline(__always)
    subscript<R>(r: (R, by: Int)) -> Self where
        R: RangeExpression, R.Bound == Int
    {
        let rRange = resolve(range: r.0, count: extents[0])
        let viewPosition = Shape.Array((rRange.lowerBound))
        let viewExtents = Shape.Array((rRange.count))
        let steps = Shape.Array((r.1))
        let (subExtents, subStrides) = makeStepped(view: viewExtents,
                                                   parent: strides,
                                                   steps: steps)
        return view(at: viewPosition, extents: subExtents, strides: subStrides)
    }
}

//==============================================================================
/// Derivative registration
extension VectorView where Self: DifferentiableTensorView {
    // https://github.com/apple/swift/blob/37b507b31c77ef969151f385cd1902dd44fb3b7f/stdlib/public/core/Array.swift#L2091
    @inlinable @inline(__always)
    func _vjpSubscript<R>(r: R) -> (value: Self, pullback: (Self) -> Self) where
        R: RangeExpression, R.Bound == Int
    {
        return (self[r], { v in
            var result = self.zeros
            result[r] = v
            return result
        })
    }
}

//==============================================================================
// Vector
public struct Vector<Element>: VectorView {
    // properties
    public let isShared: Bool
    public let format: TensorFormat = .vector
    public let shape: Shape1
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int
    
    public init(shape: Shape,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
    }
}

extension Vector: Differentiable & DifferentiableTensorView where
    Element: DifferentiableElement
{
    public typealias TangentVector = Vector
}

//******************************************************************************
//******************************************************************************
//******************************************************************************

//==============================================================================
// MatrixView protocol
public protocol MatrixView: TensorView  where Shape == Shape2 { }

public enum MatrixLayout { case rowMajor, columnMajor }

extension Matrix: Codable where Element: Codable {}

extension Matrix: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

extension Matrix: Equatable where Element: Equatable { }

extension Matrix: AdditiveArithmetic where Element: Numeric {
    public static var zero: Matrix<Element> {
        Matrix<Element>(element: Element.zero)
    }
}

//==============================================================================
// MatrixView extensions
public extension MatrixView {
    //--------------------------------------------------------------------------
    /// reserved space
    init(extents: Shape.Array, layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        self.init(extents: extents.storage, layout: layout, name: name)
    }
    
    init(extents: Shape.Tuple, layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        self = Self.create(Self.matrixShape(extents, layout), name)
    }
    
    init(_ rows: Int, _ cols: Int, layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        self.init(extents: (rows, cols), layout: layout, name: name)
    }

    //--------------------------------------------------------------------------
    /// from single `Element`
    init(element: Element, name: String? = nil) {
        let shape = Shape(extents: Shape.ones)
        self = Self.create([element], shape, name)
    }

    //--------------------------------------------------------------------------
    /// from single `AnyConvertable`
    init<T>(with element: T, name: String? = nil) where
        T: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(extents: Shape.ones)
        self = Self.create([Element(any: element)], shape, name)
    }

    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    init<C>(_ rows: Int , _ cols: Int, elements: C,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let shape = Self.matrixShape((rows, cols), layout)
        assert(shape.count == elements.count)
        self = Self.create(elements, shape, name)
    }

    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    init<C>(_ rows: Int, _ cols: Int, with elements: C,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let shape = Self.matrixShape((rows, cols), layout)
        assert(shape.count == elements.count)
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 2D `Element` collection
    init<T>(elements: [[T]], name: String? = nil) where T == Element{
        let shape = Shape(extents: (elements.count, elements.first!.count))
        self = Self.create(elements.joined(), shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 2D `AnyConvertable` collection
    init<T>(with elements: [[T]], name: String? = nil)
        where T: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(extents: (elements.count, elements.first!.count))
        let flatElements = elements.joined().lazy.map {
            Element(any: $0)
        }
        self = Self.create(flatElements, shape, name)
    }

    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ rows: Int, _ cols: Int,
         referenceTo buffer: UnsafeBufferPointer<Element>,
         layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        let shape = Self.matrixShape((rows, cols), layout)
        self = Self.create(referenceTo: buffer, shape, name)
    }

    //--------------------------------------------------------------------------
    /// with reference to read write buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ rows: Int, _ cols: Int,
         referenceTo buffer: UnsafeMutableBufferPointer<Element>,
         layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        let shape = Self.matrixShape((rows, cols), layout)
        self = Self.create(referenceTo: buffer, shape, name)
    }
    
    //--------------------------------------------------------------------------
    // typed views
    func createBoolTensor(with extents: Shape.Array) -> Matrix<Bool> {
        Matrix<Bool>(extents: extents)
    }
    
    func createIndexTensor(with extents: Shape.Array) -> Matrix<IndexElement> {
        Matrix<IndexElement>(extents: extents)
    }

    //--------------------------------------------------------------------------
    // transpose
    var t: Self {
        return Self.init(shape: shape.transposed(),
                         tensorArray: tensorArray,
                         viewOffset: viewOffset,
                         isShared: isShared)
    }
    
    //--------------------------------------------------------------------------
    // utilities
    private static func matrixShape(_ extents: Shape.Tuple,
                                    _ layout: MatrixLayout) -> Shape
    {
        let shape = Shape(extents: extents)
        return layout == .rowMajor ? shape : shape.columnMajor
    }
}

//==============================================================================
// range subscripting
public extension MatrixView {
    //--------------------------------------------------------------------------
    // TODO: probably move these off onto the TensorViewCollection
    var startIndex: MatrixIndex { return MatrixIndex(view: self, at: (0, 0)) }
    var endIndex: MatrixIndex { return MatrixIndex(endOf: self) }
    
    @inlinable @inline(__always)
    subscript<R>(r: R, c: UnboundedRange) -> Self
        where R: RangeExpression, R.Bound == Int { self[r, 0...] }

    @inlinable @inline(__always)
    subscript<R>(r: UnboundedRange, c: R) -> Self
        where R: RangeExpression, R.Bound == Int { self[0..., c] }
    
    @inlinable @inline(__always)
    subscript<R, C>(r: R, c: C) -> Self where
        R: RangeExpression, R.Bound == Int,
        C: RangeExpression, C.Bound == Int
    {
        let rRange = resolve(range: r, count: extents[0])
        let cRange = resolve(range: c, count: extents[1])
        let viewPosition = Shape.Array((rRange.lowerBound, cRange.lowerBound))
        let viewExtents = Shape.Array((rRange.count, cRange.count))
        return view(at: viewPosition, extents: viewExtents)
    }
    
    @inlinable @inline(__always)
    subscript<R, C>(r: (R, by: Int), c: (C, by: Int)) -> Self where
        R: RangeExpression, R.Bound == Int,
        C: RangeExpression, C.Bound == Int
    {
        let rRange = resolve(range: r.0, count: extents[0])
        let cRange = resolve(range: c.0, count: extents[1])
        let viewPosition = Shape.Array((rRange.lowerBound, cRange.lowerBound))
        let viewExtents = Shape.Array((rRange.count, cRange.count))
        let steps = Shape.Array((r.1, c.1))
        let (subExtents, subStrides) = makeStepped(view: viewExtents,
                                                   parent: strides,
                                                   steps: steps)
        return view(at: viewPosition, extents: subExtents, strides: subStrides)
    }
}

//==============================================================================
// Matrix
public struct Matrix<Element>: MatrixView {
    // properties
    public let isShared: Bool
    public let format: TensorFormat = .matrix
    public let shape: Shape2
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int

    public init(shape: Shape,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
    }
}

extension Matrix: Differentiable & DifferentiableTensorView where
    Element: DifferentiableElement
{
    public typealias TangentVector = Matrix
}

//******************************************************************************
//******************************************************************************
//******************************************************************************

//==============================================================================
// VolumeView protocol
public protocol VolumeView: TensorView  where Shape == Shape3 {}

extension Volume: Codable where Element: Codable {}

extension Volume: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

extension Volume: Equatable where Element: Equatable { }

extension Volume: AdditiveArithmetic where Element: Numeric {
    public static var zero: Volume<Element> {
        Volume<Element>(element: Element.zero)
    }
}

//==============================================================================
// VolumeView extensions
public extension VolumeView {
    //--------------------------------------------------------------------------
    /// reserved space
    init(extents: Shape.Array, name: String? = nil) {
        self = Self.create(Shape(extents: extents), name)
    }
    
    init(extents: Shape.Tuple, name: String? = nil) {
        self.init(extents: Shape.Array(extents), name: name)
    }

    init(_ deps: Int, _ rows: Int, _ cols: Int, name: String? = nil) {
        self.init(extents: (deps, rows, cols), name: name)
    }
    
    //--------------------------------------------------------------------------
    /// from single `Element`
    init(element: Element, name: String? = nil) {
        let shape = Shape(extents: Shape.ones)
        self = Self.create([element], shape, name)
    }

    //--------------------------------------------------------------------------
    /// from single `AnyConvertable`
    init<T>(with element: T, name: String? = nil) where
        T: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(extents: Shape.ones)
        self = Self.create([Element(any: element)], shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    init<C>(_ deps: Int, _ rows: Int, _ cols: Int,
            elements: C, name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let shape = Shape(extents: (deps, rows, cols))
        assert(shape.count == elements.count)
        self = Self.create(elements, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    init<C>(_ deps: Int, _ rows: Int, _ cols: Int,
            with elements: C, name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(extents: (deps, rows, cols))
        assert(shape.count == elements.count)
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 3D `Element` collection
    init<T>(elements: [[[T]]], name: String? = nil) where T == Element{
        let shape = Shape(extents: (elements.count,
                                    elements.first!.count,
                                    elements.first!.first!.count))
        let flatElements = elements.joined().joined()
        self = Self.create(flatElements, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 3D `AnyConvertable` collection
    init<T>(with elements: [[[T]]], name: String? = nil)
        where T: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(extents: (elements.count,
                                    elements.first!.count,
                                    elements.first!.first!.count))
        let flatElements = elements.joined().joined().lazy.map {
            Element(any: $0)
        }
        self = Self.create(flatElements, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ deps: Int, _ rows: Int, _ cols: Int,
         referenceTo buffer: UnsafeBufferPointer<Element>,
         name: String? = nil)
    {
        let shape = Shape(extents: (deps, rows, cols))
        self = Self.create(referenceTo: buffer, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read write buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ deps: Int, _ rows: Int, _ cols: Int,
         referenceTo buffer: UnsafeMutableBufferPointer<Element>,
         name: String? = nil)
    {
        let shape = Shape(extents: (deps, rows, cols))
        self = Self.create(referenceTo: buffer, shape, name)
    }
    
    //--------------------------------------------------------------------------
    // typed views
    func createBoolTensor(with extents: Shape.Array) -> Volume<Bool> {
        Volume<Bool>(extents: extents)
    }
    
    func createIndexTensor(with extents: Shape.Array) -> Volume<IndexElement> {
        Volume<IndexElement>(extents: extents)
    }
}

//==============================================================================
// range subscripting
public extension VolumeView {
    //--------------------------------------------------------------------------
    // TODO: probably move these off onto the TensorViewCollection
    var startIndex: VolumeIndex { return VolumeIndex(view: self, at: (0, 0, 0))}
    var endIndex: VolumeIndex { return VolumeIndex(endOf: self) }
    
    @inlinable @inline(__always)
    subscript<D, R, C>(d: D, r: R, c: C) -> Self where
        D: RangeExpression, D.Bound == Int,
        R: RangeExpression, R.Bound == Int,
        C: RangeExpression, C.Bound == Int
    {
        let dRange = resolve(range: d, count: extents[0])
        let rRange = resolve(range: r, count: extents[1])
        let cRange = resolve(range: c, count: extents[2])
        let viewPosition = (dRange.lowerBound,
                            rRange.lowerBound,
                            cRange.lowerBound)
        let viewExtents = (dRange.count, rRange.count, cRange.count)
        return view(at: viewPosition, extents: viewExtents)
    }
    
    @inlinable @inline(__always)
    subscript(_ d: RangeInterval, r: RangeInterval, c: RangeInterval) -> Self {
        let dRange = resolve(range: d, count: extents[0])
        let rRange = resolve(range: r, count: extents[1])
        let cRange = resolve(range: c, count: extents[2])
        let viewPosition = Shape.Array((dRange.from, rRange.from, cRange.from))
        let viewExtents = Shape.Array((dRange.to, rRange.to, cRange.to))
        let steps = Shape.Array((dRange.step, rRange.step, cRange.step))
        let (subExtents, subStrides) = makeStepped(view: viewExtents,
                                                   parent: shape.strides,
                                                   steps: steps)
        return view(at: viewPosition, extents: subExtents, strides: subStrides)
    }
}

//==============================================================================
// Volume
public struct Volume<Element>: VolumeView {
    // properties
    public let isShared: Bool
    public let format: TensorFormat = .volume
    public let shape: Shape3
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int
    
    public init(shape: Shape,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
    }
}

extension Volume: Differentiable & DifferentiableTensorView where
    Element: DifferentiableElement
{
    public typealias TangentVector = Volume
}
