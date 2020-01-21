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

extension VectorType: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

extension VectorType: Equatable where Element: Equatable { }

extension VectorType: AdditiveArithmetic where Element: Numeric {
    @inlinable @inline(__always)
    public static var zero: VectorType<Element> {
        VectorType<Element>(element: Element.zero)
    }
}

//==============================================================================
// VectorView initialization extensions
public extension VectorView {
    //--------------------------------------------------------------------------
    /// reserved space
    @inlinable @inline(__always)
    init(extents: Shape.Array, name: String? = nil) {
        self = Self.create(Shape(extents: extents), name)
    }
    
    @inlinable @inline(__always)
    init(extents: Shape.Tuple, name: String? = nil) {
        self.init(extents: Shape.Array(extents), name: name)
    }
    
    @inlinable @inline(__always)
    init(count: Int, name: String? = nil) {
        self.init(extents: (count), name: name)
    }
    
    //--------------------------------------------------------------------------
    /// from single `Element`
    @inlinable @inline(__always)
    init(element: Element, name: String? = nil) {
        self = Self.create([element], Shape(extents: (1)), name)
    }
    
    //--------------------------------------------------------------------------
    /// from single `AnyConvertable`
    @inlinable @inline(__always)
    init<T>(with element: T, name: String? = nil) where
        T: AnyConvertable, Element: AnyConvertable
    {
        self = Self.create([Element(any: element)], Shape(extents: (1)), name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    @inlinable @inline(__always)
    init<C>(elements: C, name: String? = nil) where
        C: Collection, C.Element == Element
    {
        self = Self.create(elements, Shape(extents: (elements.count)), name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    @inlinable @inline(__always)
    init<C>(with elements: C, name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self = Self.create(elements.lazy.map { Element(any: $0) },
                           Shape(extents: (elements.count)), name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    @inlinable @inline(__always)
    init(referenceTo buffer: UnsafeBufferPointer<Element>, name: String? = nil)
    {
        let shape = Shape(extents: (buffer.count))
        self = Self.create(referenceTo: buffer, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read write buffer
    /// useful for memory mapped databases, or hardware device buffers
    @inlinable @inline(__always)
    init(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
         name: String? = nil)
    {
        let shape = Shape(extents: (buffer.count))
        self = Self.create(referenceTo: buffer, shape, name)
    }

    //--------------------------------------------------------------------------
    // typed views
    @inlinable @inline(__always)
    func createBoolTensor(with extents: Shape.Array) -> VectorType<Bool> {
        VectorType<Bool>(extents: extents)
    }
    
    @inlinable @inline(__always)
    func createIndexTensor(with extents: Shape.Array) -> VectorType<IndexType> {
        VectorType<IndexType>(extents: extents)
    }
}

//==============================================================================
// VectorView subscripting and collections
public extension VectorView {
    //--------------------------------------------------------------------------
    // TODO: probably move these off onto the TensorViewCollection
    @inlinable @inline(__always)
    var startIndex: VectorIndex { VectorIndex(view: self, at: Shape.zeros.tuple)}

    @inlinable @inline(__always)
    var endIndex: VectorIndex { VectorIndex(endOf: self) }

    //--------------------------------------------------------------------------
    /// Swift array of elements
    @inlinable @inline(__always)
    var array: [Element] { [Element](elements()) }

    //--------------------------------------------------------------------------
    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript(index: Int) -> Self {
        get { self[(index), (index + 1), Shape.ones.tuple] }
        set { self[(index), (index + 1), Shape.ones.tuple] = newValue }
    }
    
    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript(r: UnboundedRange) -> Self { self }

    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(range: R) -> Self
        where R: PartialRangeExpression, R.Bound == Int
    {
        get {
            let r = withoutDerivative(at: range.relativeTo(0..<extents[0]))
            return self[(r.start), (r.end), (r.step)]
        }
        set {
            let r = withoutDerivative(at: range.relativeTo(0..<extents[0]))
            self[(r.start), (r.end), (r.step)] = newValue
        }
    }
}

//==============================================================================
// VectorType
public struct VectorType<Element>: VectorView
    where Element: TensorElementConformance
{
    // properties
    public let isShared: Bool
    public let format: TensorFormat
    public let shape: Shape1
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int
    
    @inlinable @inline(__always)
    public init(shape: Shape1,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
        self.format = .vector
    }
}

extension VectorType: Differentiable & DifferentiableTensorView where
    Element: DifferentiableElement
{
    public typealias TangentVector = VectorType
}

//******************************************************************************
//******************************************************************************
//******************************************************************************

//==============================================================================
// MatrixView protocol
public protocol MatrixView: TensorView  where Shape == Shape2 { }

public enum MatrixLayout { case rowMajor, columnMajor }

extension MatrixType: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

extension MatrixType: Equatable where Element: Equatable { }

extension MatrixType: AdditiveArithmetic where Element: Numeric {
    @inlinable @inline(__always)
    public static var zero: MatrixType<Element> {
        MatrixType<Element>(element: Element.zero)
    }
}

//==============================================================================
// MatrixView initialization extensions
public extension MatrixView {
    //--------------------------------------------------------------------------
    /// reserved space
    @inlinable @inline(__always)
    init(extents: Shape.Array, layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        self.init(extents: extents.storage, layout: layout, name: name)
    }
    
    @inlinable @inline(__always)
    init(extents: Shape.Tuple, layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        self = Self.create(Self.matrixShape(extents, layout), name)
    }
    
    @inlinable @inline(__always)
    init(_ rows: Int, _ cols: Int, layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        self.init(extents: (rows, cols), layout: layout, name: name)
    }

    //--------------------------------------------------------------------------
    /// from single `Element`
    @inlinable @inline(__always)
    init(element: Element, name: String? = nil) {
        let shape = Shape(extents: Shape.ones)
        self = Self.create([element], shape, name)
    }

    //--------------------------------------------------------------------------
    /// from single `AnyConvertable`
    @inlinable @inline(__always)
    init<T>(with element: T, name: String? = nil) where
        T: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(extents: Shape.ones)
        self = Self.create([Element(any: element)], shape, name)
    }

    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    @inlinable @inline(__always)
    init<C>(_ rows: Int , _ cols: Int, elements: C,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let shape = Self.matrixShape((rows, cols), layout)
        assert(shape.count == elements.count, _messageElementCountMismatch)
        self = Self.create(elements, shape, name)
    }

    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    @inlinable @inline(__always)
    init<C>(_ rows: Int, _ cols: Int, with elements: C,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let shape = Self.matrixShape((rows, cols), layout)
        assert(shape.count == elements.count, _messageElementCountMismatch)
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 2D `Element` collection
    @inlinable @inline(__always)
    init<T>(elements: [[T]], name: String? = nil) where T == Element{
        let shape = Shape(extents: (elements.count, elements.first!.count))
        self = Self.create(elements.joined(), shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 2D `AnyConvertable` collection
    @inlinable @inline(__always)
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
    @inlinable @inline(__always)
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
    @inlinable @inline(__always)
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
    @inlinable @inline(__always)
    func createBoolTensor(with extents: Shape.Array) -> MatrixType<Bool> {
        MatrixType<Bool>(extents: extents)
    }
    
    @inlinable @inline(__always)
    func createIndexTensor(with extents: Shape.Array) -> MatrixType<IndexType> {
        MatrixType<IndexType>(extents: extents)
    }

    //--------------------------------------------------------------------------
    // transpose
    @inlinable @inline(__always)
    var t: Self {
        Self.init(shape: shape.transposed(),
                  tensorArray: tensorArray,
                  viewOffset: viewOffset,
                  isShared: isShared)
    }
    
    //--------------------------------------------------------------------------
    // utilities
    @inlinable @inline(__always)
    static func matrixShape(_ extents: Shape.Tuple,
                            _ layout: MatrixLayout) -> Shape
    {
        let shape = Shape(extents: extents)
        return layout == .rowMajor ? shape : shape.columnMajor
    }
}

//==============================================================================
// MatrixView collection extensions
public extension MatrixView {
    //--------------------------------------------------------------------------
    // TODO: probably move these off onto the TensorViewCollection
    @inlinable @inline(__always)
    var startIndex: MatrixIndex { MatrixIndex(view: self, at: Shape.zeros.tuple)}

    @inlinable @inline(__always)
    var endIndex: MatrixIndex { MatrixIndex(endOf: self) }
    
    //--------------------------------------------------------------------------
    /// Swift array of elements
    @inlinable @inline(__always)
    var array: [[Element]] {
        var result = [[Element]]()
        for row in 0..<extents[0] {
            result.append([Element](self[row..|1, ...].elements()))
        }
        return result
    }

    //--------------------------------------------------------------------------
    // single element
    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript(r: Int, c: Int) -> Self {
        get { self[(r, c), (r + 1, c + 1), Shape.ones.tuple] }
        set { self[(r, c), (r + 1, c + 1), Shape.ones.tuple] = newValue }
    }

    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript<R, C>(rows: R, cols: C) -> Self where
        R: PartialRangeExpression, R.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int
    {
        get {
            let r = withoutDerivative(at: rows.relativeTo(0..<extents[0]))
            let c = withoutDerivative(at: cols.relativeTo(0..<extents[1]))
            return self[(r.start, c.start), (r.end, c.end), (r.step, c.step)]
        }
        
        set {
            let r = withoutDerivative(at: rows.relativeTo(0..<extents[0]))
            let c = withoutDerivative(at: cols.relativeTo(0..<extents[1]))
            self[(r.start, c.start), (r.end, c.end), (r.step, c.step)] = newValue
        }
    }
    
    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(rows: R, cols: UnboundedRange) -> Self
        where R: PartialRangeExpression, R.Bound == Int {
        get { self[rows, 0...] }
        set { self[rows, 0...] = newValue }
    }
    
    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript<C>(rows: UnboundedRange, cols: C) -> Self
        where C: PartialRangeExpression, C.Bound == Int {
        get { self[0..., cols] }
        set { self[0..., cols] = newValue }
    }
}

//==============================================================================
// MatrixType
public struct MatrixType<Element>: MatrixView
    where Element: TensorElementConformance
{
    // properties
    public let isShared: Bool
    public let format: TensorFormat
    public let shape: Shape2
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int

    @inlinable @inline(__always)
    public init(shape: Shape2,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
        self.format = .matrix
    }
}

extension MatrixType: Differentiable & DifferentiableTensorView where
    Element: DifferentiableElement
{
    public typealias TangentVector = MatrixType
}

//******************************************************************************
//******************************************************************************
//******************************************************************************

//==============================================================================
// VolumeView protocol
public protocol VolumeView: TensorView  where Shape == Shape3 {}

extension VolumeType: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

extension VolumeType: Equatable where Element: Equatable { }

extension VolumeType: AdditiveArithmetic where Element: Numeric {
    @inlinable @inline(__always)
    public static var zero: VolumeType<Element> {
        VolumeType<Element>(element: Element.zero)
    }
}

//==============================================================================
// VolumeView extensions
public extension VolumeView {
    //--------------------------------------------------------------------------
    /// reserved space
    @inlinable @inline(__always)
    init(extents: Shape.Array, name: String? = nil) {
        self = Self.create(Shape(extents: extents), name)
    }
    
    @inlinable @inline(__always)
    init(extents: Shape.Tuple, name: String? = nil) {
        self.init(extents: Shape.Array(extents), name: name)
    }

    @inlinable @inline(__always)
    init(_ deps: Int, _ rows: Int, _ cols: Int, name: String? = nil) {
        self.init(extents: (deps, rows, cols), name: name)
    }
    
    //--------------------------------------------------------------------------
    /// from single `Element`
    @inlinable @inline(__always)
    init(element: Element, name: String? = nil) {
        let shape = Shape(extents: Shape.ones)
        self = Self.create([element], shape, name)
    }

    //--------------------------------------------------------------------------
    /// from single `AnyConvertable`
    @inlinable @inline(__always)
    init<T>(with element: T, name: String? = nil) where
        T: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(extents: Shape.ones)
        self = Self.create([Element(any: element)], shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    @inlinable @inline(__always)
    init<C>(_ deps: Int, _ rows: Int, _ cols: Int,
            elements: C, name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let shape = Shape(extents: (deps, rows, cols))
        assert(shape.count == elements.count, _messageElementCountMismatch)
        self = Self.create(elements, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    @inlinable @inline(__always)
    init<C>(_ deps: Int, _ rows: Int, _ cols: Int,
            with elements: C, name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(extents: (deps, rows, cols))
        assert(shape.count == elements.count, _messageElementCountMismatch)
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 3D `Element` collection
    @inlinable @inline(__always)
    init<T>(elements: [[[T]]], name: String? = nil) where T == Element{
        let shape = Shape(extents: (elements.count,
                                    elements.first!.count,
                                    elements.first!.first!.count))
        let flatElements = elements.joined().joined()
        self = Self.create(flatElements, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 3D `AnyConvertable` collection
    @inlinable @inline(__always)
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
    @inlinable @inline(__always)
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
    @inlinable @inline(__always)
    init(_ deps: Int, _ rows: Int, _ cols: Int,
         referenceTo buffer: UnsafeMutableBufferPointer<Element>,
         name: String? = nil)
    {
        let shape = Shape(extents: (deps, rows, cols))
        self = Self.create(referenceTo: buffer, shape, name)
    }
    
    //--------------------------------------------------------------------------
    // typed views
    @inlinable @inline(__always)
    func createBoolTensor(with extents: Shape.Array) -> VolumeType<Bool> {
        VolumeType<Bool>(extents: extents)
    }
    
    @inlinable @inline(__always)
    func createIndexTensor(with extents: Shape.Array) -> VolumeType<IndexType> {
        VolumeType<IndexType>(extents: extents)
    }
}

//==============================================================================
// MatrixView extensions
public extension VolumeView {
    //--------------------------------------------------------------------------
    // TODO: probably move these off onto the TensorViewCollection
    @inlinable @inline(__always)
    var startIndex: VolumeIndex {
        VolumeIndex(view: self, at: Shape.zeros.tuple)
    }

    @inlinable @inline(__always)
    var endIndex: VolumeIndex { VolumeIndex(endOf: self) }
    
    //--------------------------------------------------------------------------
    /// Swift array of elements
    @inlinable @inline(__always)
    var array: [[[Element]]] {
        var result = [[[Element]]]()
        for di in 0..<extents[0] {
            var depth = [[Element]]()
            for ri in 0..<extents[1] {
                depth.append([Element](self[di..|1, ri..|1, ...].elements()))
            }
            result.append(depth)
        }
        return result
    }

    //--------------------------------------------------------------------------
    // single element
    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript(d: Int, r: Int, c: Int) -> Self {
        get { self[(d, r, c), (d + 1, r + 1, c + 1), Shape.ones.tuple] }
        set { self[(d, r, c), (d + 1, r + 1, c + 1), Shape.ones.tuple] =
            newValue }
    }
    
    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, R, C>(deps: D, rows: R, cols: C) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        R: PartialRangeExpression, R.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int
        {
        get {
            let d = withoutDerivative(at: deps.relativeTo(0..<extents[0]))
            let r = withoutDerivative(at: rows.relativeTo(0..<extents[1]))
            let c = withoutDerivative(at: cols.relativeTo(0..<extents[2]))
            return self[(d.start, r.start, c.start),
                        (d.end, r.end, c.end),
                        (d.step, r.step, c.step)]
        }
        
        set {
            let d = withoutDerivative(at: deps.relativeTo(0..<extents[0]))
            let r = withoutDerivative(at: rows.relativeTo(0..<extents[1]))
            let c = withoutDerivative(at: cols.relativeTo(0..<extents[2]))
            self[(d.start, r.start, c.start),
                 (d.end, r.end, c.end),
                 (d.step, r.step, c.step)] = newValue
        }
    }
    
    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript<D>(deps: D, rows: UnboundedRange, cols: UnboundedRange) -> Self
        where D: PartialRangeExpression, D.Bound == Int {
        get { self[deps, 0..., 0...] }
        set { self[deps, 0..., 0...] = newValue }
    }
    
    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, R>(deps: D, rows: R, cols: UnboundedRange) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        R: PartialRangeExpression, R.Bound == Int {
        get { self[deps, rows, 0...] }
        set { self[deps, rows, 0...] = newValue }
    }
    
    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, C>(deps: D, rows: UnboundedRange, cols: C) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int {
        get { self[deps, 0..., cols] }
        set { self[deps, 0..., cols] = newValue }
    }

    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(deps: UnboundedRange, rows: R, cols: UnboundedRange) -> Self
        where R: PartialRangeExpression, R.Bound == Int {
        get { self[0..., rows, 0...] }
        set { self[0..., rows, 0...] = newValue }
    }
    
    @inlinable @inline(__always)
    @differentiable(where Self: DifferentiableTensorView)
    subscript<C>(deps: UnboundedRange, rows: UnboundedRange, cols: C) -> Self
        where C: PartialRangeExpression, C.Bound == Int {
        get { self[0..., 0..., cols] }
        set { self[0..., 0..., cols] = newValue }
    }
}

//==============================================================================
// VolumeType
public struct VolumeType<Element>: VolumeView
    where Element: TensorElementConformance
{
    // properties
    public let isShared: Bool
    public let format: TensorFormat
    public let shape: Shape3
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int
    
    @inlinable @inline(__always)
    public init(shape: Shape3,
                tensorArray: TensorArray<Element>,
                viewOffset: Int,
                isShared: Bool)
    {
        self.shape = shape
        self.tensorArray = tensorArray
        self.viewOffset = viewOffset
        self.isShared = isShared
        self.format = .volume
    }
}

extension VolumeType: Differentiable & DifferentiableTensorView where
    Element: DifferentiableElement
{
    public typealias TangentVector = VolumeType
}
