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

extension VectorT: Codable where Element: Codable {}

extension VectorT: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

extension VectorT: Equatable where Element: Equatable { }

extension VectorT: AdditiveArithmetic where Element: Numeric {
    public static var zero: VectorT<Element> {
        VectorT<Element>(element: Element.zero)
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
    func createBoolTensor(with extents: Shape.Array) -> VectorT<Bool> {
        VectorT<Bool>(extents: extents)
    }
    
    func createIndexTensor(with extents: Shape.Array) -> VectorT<IndexT> {
        VectorT<IndexT>(extents: extents)
    }
}

//==============================================================================
// VectorView subscripting
public extension VectorView {
    //--------------------------------------------------------------------------
    // TODO: probably move these off onto the TensorViewCollection
    var startIndex: VectorIndex { VectorIndex(view: self, at: Shape.zeros.tuple)}
    var endIndex: VectorIndex { VectorIndex(endOf: self) }

    //--------------------------------------------------------------------------
    @inlinable @inline(__always)
    subscript(index: Int) -> Self {
        get { self[(index), (index + 1), Shape.ones.tuple] }
        set { self[(index), (index + 1), Shape.ones.tuple] = newValue }
    }
    
    subscript(r: UnboundedRange) -> Self { self }

    // TODO
//    @differentiable(vjp: vjpSubscript where Self: DifferentiableTensorView)
    @inlinable @inline(__always)
    subscript<R>(range: R) -> Self
        where R: StridedRangeExpression, R.Bound == Int
    {
        get {
            let r = range.stridedRangeRelative(to: 0..<extents[0])
            return self[(r.from), (r.to), (r.by)]
        }
        set {
            let r = range.stridedRangeRelative(to: 0..<extents[0])
            self[(r.from), (r.to), (r.by)] = newValue
        }
    }
}

//==============================================================================
// Vector
public struct VectorT<Element>: VectorView {
    // properties
    public let isShared: Bool
    public let format: TensorFormat = .vector
    public let shape: Shape1
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int
    
    public init(shape: Shape1,
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

extension VectorT: Differentiable & DifferentiableTensorView where
    Element: DifferentiableElement
{
    public typealias TangentVector = VectorT
}

//******************************************************************************
//******************************************************************************
//******************************************************************************

//==============================================================================
// MatrixView protocol
public protocol MatrixView: TensorView  where Shape == Shape2 { }

public enum MatrixLayout { case rowMajor, columnMajor }

extension MatrixT: Codable where Element: Codable {}

extension MatrixT: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

extension MatrixT: Equatable where Element: Equatable { }

extension MatrixT: AdditiveArithmetic where Element: Numeric {
    public static var zero: MatrixT<Element> {
        MatrixT<Element>(element: Element.zero)
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
    func createBoolTensor(with extents: Shape.Array) -> MatrixT<Bool> {
        MatrixT<Bool>(extents: extents)
    }
    
    func createIndexTensor(with extents: Shape.Array) -> MatrixT<IndexT> {
        MatrixT<IndexT>(extents: extents)
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
// MatrixView extensions
public extension MatrixView {
    //--------------------------------------------------------------------------
    // TODO: probably move these off onto the TensorViewCollection
    var startIndex: MatrixIndex { MatrixIndex(view: self, at: Shape.ones.tuple)}
    var endIndex: MatrixIndex { MatrixIndex(endOf: self) }
    
    //--------------------------------------------------------------------------
    // single element
    @inlinable @inline(__always)
    subscript(r: Int, c: Int) -> Self {
        get { self[(r, c), (r + 1, c + 1), Shape.ones.tuple] }
        set { self[(r, c), (r + 1, c + 1), Shape.ones.tuple] = newValue }
    }

    // TODO(TF-281): Rewrite `@differentiable` attribute as a `@differentiating`
    // attribute when `@differentiating` supports subscript declaration
    // references.
    //    @differentiable(vjp: vjpSubscript where Self: DifferentiableTensorView)
    @inlinable @inline(__always)
    subscript<R, C>(rows: R, cols: C) -> Self where
        R: StridedRangeExpression, R.Bound == Int,
        C: StridedRangeExpression, C.Bound == Int
    {
        get {
            let r = rows.stridedRangeRelative(to: 0..<extents[0])
            let c = cols.stridedRangeRelative(to: 0..<extents[1])
            return self[(r.from, c.from), (r.to, c.to), (r.by, c.by)]
        }
        
        set {
            let r = rows.stridedRangeRelative(to: 0..<extents[0])
            let c = cols.stridedRangeRelative(to: 0..<extents[1])
            self[(r.from, c.from), (r.to, c.to), (r.by, c.by)] = newValue
        }
    }
    
    subscript<R>(rows: R, cols: UnboundedRange) -> Self
        where R: StridedRangeExpression, R.Bound == Int {
        get { self[rows, 0...] }
        set { self[rows, 0...] = newValue }
    }
    
    subscript<C>(rows: UnboundedRange, cols: C) -> Self
        where C: StridedRangeExpression, C.Bound == Int {
        get { self[0..., cols] }
        set { self[0..., cols] = newValue }
    }
}

//==============================================================================
// Matrix
public struct MatrixT<Element>: MatrixView {
    // properties
    public let isShared: Bool
    public let format: TensorFormat = .matrix
    public let shape: Shape2
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int

    public init(shape: Shape2,
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

extension MatrixT: Differentiable & DifferentiableTensorView where
    Element: DifferentiableElement
{
    public typealias TangentVector = MatrixT
}

//******************************************************************************
//******************************************************************************
//******************************************************************************

//==============================================================================
// VolumeView protocol
public protocol VolumeView: TensorView  where Shape == Shape3 {}

extension VolumeT: Codable where Element: Codable {}

extension VolumeT: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

extension VolumeT: Equatable where Element: Equatable { }

extension VolumeT: AdditiveArithmetic where Element: Numeric {
    public static var zero: VolumeT<Element> {
        VolumeT<Element>(element: Element.zero)
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
    func createBoolTensor(with extents: Shape.Array) -> VolumeT<Bool> {
        VolumeT<Bool>(extents: extents)
    }
    
    func createIndexTensor(with extents: Shape.Array) -> VolumeT<IndexT> {
        VolumeT<IndexT>(extents: extents)
    }
}

//==============================================================================
// MatrixView extensions
public extension VolumeView {
    //--------------------------------------------------------------------------
    // TODO: probably move these off onto the TensorViewCollection
    var startIndex: VolumeIndex { VolumeIndex(view: self,
                                              at: Shape.zeros.tuple) }
    var endIndex: VolumeIndex { VolumeIndex(endOf: self) }
    
    //--------------------------------------------------------------------------
    // single element
    @inlinable @inline(__always)
    subscript(d: Int, r: Int, c: Int) -> Self {
        get { self[(d, r, c), (d + 1, r + 1, c + 1), Shape.ones.tuple] }
        set { self[(d, r, c), (d + 1, r + 1, c + 1), Shape.ones.tuple] =
            newValue }
    }
    
    // TODO(TF-281): Rewrite `@differentiable` attribute as a `@differentiating`
    // attribute when `@differentiating` supports subscript declaration
    // references.
    //    @differentiable(vjp: vjpSubscript where Self: DifferentiableTensorView)
    @inlinable @inline(__always)
    subscript<D, R, C>(deps: D, rows: R, cols: C) -> Self where
        D: StridedRangeExpression, D.Bound == Int,
        R: StridedRangeExpression, R.Bound == Int,
        C: StridedRangeExpression, C.Bound == Int
        {
        get {
            let d = deps.stridedRangeRelative(to: 0..<extents[0])
            let r = rows.stridedRangeRelative(to: 0..<extents[1])
            let c = cols.stridedRangeRelative(to: 0..<extents[2])
            return self[(d.from, r.from, c.from),
                        (d.to, r.to, c.to),
                        (d.by, r.by, c.by)]
        }
        
        set {
            let d = deps.stridedRangeRelative(to: 0..<extents[0])
            let r = rows.stridedRangeRelative(to: 0..<extents[1])
            let c = cols.stridedRangeRelative(to: 0..<extents[2])
            self[(d.from, r.from, c.from),
                 (d.to, r.to, c.to),
                 (d.by, r.by, c.by)] = newValue
        }
    }
    
    subscript<D>(deps: D, rows: UnboundedRange, cols: UnboundedRange) -> Self
        where D: StridedRangeExpression, D.Bound == Int {
        get { self[deps, 0..., 0...] }
        set { self[deps, 0..., 0...] = newValue }
    }
    
    subscript<D, R>(deps: D, rows: R, cols: UnboundedRange) -> Self where
        D: StridedRangeExpression, D.Bound == Int,
        R: StridedRangeExpression, R.Bound == Int {
        get { self[deps, rows, 0...] }
        set { self[deps, rows, 0...] = newValue }
    }
    
    subscript<D, C>(deps: D, rows: UnboundedRange, cols: C) -> Self where
        D: StridedRangeExpression, D.Bound == Int,
        C: StridedRangeExpression, C.Bound == Int {
        get { self[deps, 0..., cols] }
        set { self[deps, 0..., cols] = newValue }
    }

    subscript<R>(deps: UnboundedRange, rows: R, cols: UnboundedRange) -> Self
        where R: StridedRangeExpression, R.Bound == Int {
        get { self[0..., rows, 0...] }
        set { self[0..., rows, 0...] = newValue }
    }
    
    subscript<C>(deps: UnboundedRange, rows: UnboundedRange, cols: C) -> Self
        where C: StridedRangeExpression, C.Bound == Int {
        get { self[0..., 0..., cols] }
        set { self[0..., 0..., cols] = newValue }
    }
}

//==============================================================================
// Volume
public struct VolumeT<Element>: VolumeView {
    // properties
    public let isShared: Bool
    public let format: TensorFormat = .volume
    public let shape: Shape3
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int
    
    public init(shape: Shape3,
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

extension VolumeT: Differentiable & DifferentiableTensorView where
    Element: DifferentiableElement
{
    public typealias TangentVector = VolumeT
}
