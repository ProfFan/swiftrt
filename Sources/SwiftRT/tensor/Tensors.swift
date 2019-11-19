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
public protocol VectorView: TensorView {}

extension Vector: Codable where Element: Codable {}

extension Vector: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// MatrixView extensions
public extension VectorView {
    //--------------------------------------------------------------------------
    /// reserved space
    init(extents: [Int], name: String? = nil) {
        self = Self.create(DataShape(extents: extents), name)
    }
    
    init(_ count: Int, name: String? = nil) {
        self.init(extents: [count], name: name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    init<C>(elements: C, name: String? = nil) where
        C: Collection, C.Element == Element
    {
        self = Self.create(elements, DataShape(extents: [elements.count]), name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    init<C>(with elements: C, name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self = Self.create(elements.lazy.map { Element(any: $0) },
                           DataShape(extents: [elements.count]),
                           name)
    }
    
    //----------------------------------
    /// repeating an Element
    init(repeating element: Element, count: Int, name: String? = nil) {
        let shape = DataShape(extents: [1]).repeated(to: [count])
        self = Self.create([element], shape, name)
    }
    
    //----------------------------------
    /// repeating any Element
    init<T>(repeating element: T, count: Int, name: String? = nil) where
        T: AnyConvertable, Element: AnyConvertable
    {
        let shape = DataShape(extents: [1]).repeated(to: [count])
        self = Self.create([Element(any: element)], shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(referenceTo buffer: UnsafeBufferPointer<Element>, name: String? = nil)
    {
        let shape = DataShape(extents: [buffer.count])
        self = Self.create(referenceTo: buffer, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read write buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
         name: String? = nil)
    {
        let shape = DataShape(extents: [buffer.count])
        self = Self.create(referenceTo: buffer, shape, name)
    }
    
    
    //--------------------------------------------------------------------------
    // typed views
    func createBoolTensor(with extents: [Int]) -> Vector<Bool> {
        Vector<Bool>(extents: extents)
    }
    
    func createIndexTensor(with extents: [Int]) -> Vector<IndexElement> {
        Vector<IndexElement>(extents: extents)
    }
    
    //--------------------------------------------------------------------------
    // utilities
    private static func matrixShape(
        _ extents: [Int],
        _ layout: MatrixLayout) -> DataShape
    {
        let shape = DataShape(extents: extents)
        return layout == .rowMajor ? shape : shape.columnMajor()
    }
    
    private static func matrixRepeatedShape(
        _ extents: [Int],
        _ repeatedExtents: [Int],
        _ layout: MatrixLayout) -> DataShape
    {
        let shape = DataShape(extents: extents).repeated(to: repeatedExtents)
        return layout == .rowMajor ? shape : shape.columnMajor()
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
    
    @inlinable @inline(__always)
    subscript<R>(r: R) -> Self where
        R: RangeExpression, R.Bound == Int
    {
        let rRange = makePositive(range: r, count: extents[0])
        return view(at: [rRange.lowerBound], extents: [rRange.count])
    }
    
    @inlinable @inline(__always)
    subscript<R>(r: (R, by: Int)) -> Self where
        R: RangeExpression, R.Bound == Int
    {
        let rRange = makePositive(range: r.0, count: extents[0])
        let viewPosition = [rRange.lowerBound]
        let viewExtents = [rRange.count]
        let steps = [r.1]
        let (subExtents, subStrides) = makeStepped(view: viewExtents,
                                                   parent: shape.strides,
                                                   steps: steps)
        return view(at: viewPosition, extents: subExtents, strides: subStrides)
    }
}

//==============================================================================
// Vector
public struct Vector<Element>: VectorView {
    // properties
    public let isShared: Bool
    public let format: TensorFormat = .vector
    public let shape: DataShape
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int
    public let singleElementExtents = [1]
    
    public init(shape: DataShape,
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

//******************************************************************************
//******************************************************************************
//******************************************************************************

//==============================================================================
// MatrixView protocol
public protocol MatrixView: TensorView {}

public enum MatrixLayout { case rowMajor, columnMajor }

extension Matrix: Codable where Element: Codable {}

extension Matrix: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// MatrixView extensions
public extension MatrixView {
    //--------------------------------------------------------------------------
    /// reserved space
    init(extents: [Int], layout: MatrixLayout = .rowMajor, name: String? = nil)
    {
        self = Self.create(Self.matrixShape(extents, layout), name)
    }
    
    init(_ rows: Int, _ cols: Int, layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        self.init(extents: [rows, cols], layout: layout, name: name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    init<C>(_ rows: Int = 1, _ cols: Int = 1, elements: C,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let shape = Self.matrixShape([rows, cols], layout)
        self = Self.create(elements, shape, name)
    }

    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    init<C>(_ rows: Int = 1, _ cols: Int = 1, with elements: C,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let shape = Self.matrixShape([rows, cols], layout)
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //----------------------------------
    /// repeating an Element
    init(repeating element: Element, rows: Int = 1, cols: Int = 1,
         layout: MatrixLayout = .rowMajor, name: String? = nil)
    {
        let shape = Self.matrixRepeatedShape([1, 1], [rows, cols], layout)
        self = Self.create([element], shape, name)
    }
    
    //----------------------------------
    /// repeating a row of `Element`
    init<C>(repeatingRow elements: C, count: Int,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let cols = elements.count
        let shape = Self.matrixRepeatedShape([1, cols], [count, cols], layout)
        self = Self.create(elements, shape, name)
    }
    
    //----------------------------------
    /// repeating a column `Element`
    init<C>(repeatingCol elements: C, count: Int,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let rows = elements.count
        let shape = Self.matrixRepeatedShape([rows, 1], [rows, count], layout)
        self = Self.create(elements, shape, name)
    }
    
    //----------------------------------
    /// repeating a row of `AnyConvertable`
    init<C>(repeatingRow elements: C, count: Int,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let cols = elements.count
        let shape = Self.matrixRepeatedShape([1, cols], [count, cols], layout)
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //----------------------------------
    /// repeating a column `AnyConvertable`
    init<C>(repeatingCol elements: C, count: Int,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let rows = elements.count
        let shape = Self.matrixRepeatedShape([rows, 1], [rows, count], layout)
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(referenceTo buffer: UnsafeBufferPointer<Element>,
         rows: Int, cols: Int,
         layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        let shape = Self.matrixShape([rows, cols], layout)
        self = Self.create(referenceTo: buffer, shape, name)
    }

    //--------------------------------------------------------------------------
    /// with reference to read write buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
         rows: Int, cols: Int,
         layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        let shape = Self.matrixShape([rows, cols], layout)
        self = Self.create(referenceTo: buffer, shape, name)
    }

    
    //--------------------------------------------------------------------------
    // typed views
    func createBoolTensor(with extents: [Int]) -> Matrix<Bool> {
        Matrix<Bool>(extents: extents)
    }
    
    func createIndexTensor(with extents: [Int]) -> Matrix<IndexElement> {
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
    private static func matrixShape(
        _ extents: [Int],
        _ layout: MatrixLayout) -> DataShape
    {
        let shape = DataShape(extents: extents)
        return layout == .rowMajor ? shape : shape.columnMajor()
    }

    private static func matrixRepeatedShape(
        _ extents: [Int],
        _ repeatedExtents: [Int],
        _ layout: MatrixLayout) -> DataShape
    {
        let shape = DataShape(extents: extents).repeated(to: repeatedExtents)
        return layout == .rowMajor ? shape : shape.columnMajor()
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
        let rRange = makePositive(range: r, count: extents[0])
        let cRange = makePositive(range: c, count: extents[1])
        let viewPosition = [rRange.lowerBound, cRange.lowerBound]
        let viewExtents = [rRange.count, cRange.count]
        return view(at: viewPosition, extents: viewExtents)
    }
    
    @inlinable @inline(__always)
    subscript<R, C>(r: (R, by: Int), c: (C, by: Int)) -> Self where
        R: RangeExpression, R.Bound == Int,
        C: RangeExpression, C.Bound == Int
    {
        let rRange = makePositive(range: r.0, count: extents[0])
        let cRange = makePositive(range: c.0, count: extents[1])
        let viewPosition = [rRange.lowerBound, cRange.lowerBound]
        let viewExtents = [rRange.count, cRange.count]
        let steps = [r.1, c.1]
        let (subExtents, subStrides) = makeStepped(view: viewExtents,
                                                   parent: shape.strides,
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
    public let shape: DataShape
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int
    public let singleElementExtents = [1, 1]

    public init(shape: DataShape,
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

//******************************************************************************
//******************************************************************************
//******************************************************************************

//==============================================================================
// VolumeView protocol
public protocol VolumeView: TensorView {}

extension Volume: Codable where Element: Codable {}

extension Volume: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// VolumeView extensions
public extension VolumeView {
    //--------------------------------------------------------------------------
    /// reserved space
    init(extents: [Int], name: String? = nil) {
        self = Self.create(DataShape(extents: extents), name)
    }
    
    init(_ deps: Int, _ rows: Int, _ cols: Int, name: String? = nil) {
        self.init(extents: [deps, rows, cols], name: name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    init<C>(_ deps: Int = 1, _ rows: Int = 1, _ cols: Int?, elements: C,
            name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let cols = cols ?? elements.count
        let shape = DataShape(extents: [deps, rows, cols])
        self = Self.create(elements, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    init<C>(_ deps: Int = 1, _ rows: Int = 1, _ cols: Int?, with elements: C,
            name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let cols = cols ?? elements.count
        let shape = DataShape(extents: [deps, rows, cols])
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //----------------------------------
    /// repeating an Element
    init(repeating element: Element,
         deps: Int = 1, rows: Int = 1, cols: Int = 1,
         name: String? = nil)
    {
        let shape = DataShape(extents: [1, 1, 1])
            .repeated(to: [deps, rows, cols])
        self = Self.create([element], shape, name)
    }
    
    //----------------------------------
    /// repeating a row of `Element`
    init<C>(repeating elements: C, deps: Int, rows: Int,
            name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let cols = elements.count
        let shape = DataShape(extents: [1, 1, cols])
            .repeated(to: [deps, rows, cols])
        self = Self.create(elements, shape, name)
    }
    
    //----------------------------------
    /// repeating a column `Element`
    init<C>(repeating elements: C, deps: Int, cols: Int,
            name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let rows = elements.count
        let shape = DataShape(extents: [1, rows, 1])
            .repeated(to: [deps, rows, cols])
        self = Self.create(elements, shape, name)
    }
    
    //----------------------------------
    /// repeating a row of `AnyConvertable`
    init<C>(repeating elements: C, deps: Int, name: String? = nil) where
        C: Collection, C.Element: Collection,
        C.Element.Element == Element
    {
        let rows = elements.count
        let cols = elements[elements.startIndex].count
        let shape = DataShape(extents: [1, rows, cols])
            .repeated(to: [deps, rows, cols])
        self = Self.create(elements.joined(), shape, name)
    }

    //----------------------------------
    /// repeating a row of `AnyConvertable`
    init<C>(repeating elements: C, deps: Int, rows: Int,
            name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let cols = elements.count
        let shape = DataShape(extents: [1, 1, cols])
            .repeated(to: [deps, rows, cols])
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //----------------------------------
    /// repeating a column `AnyConvertable`
    init<C>(repeating elements: C, deps: Int, cols: Int,
            name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let rows = elements.count
        let shape = DataShape(extents: [1, rows, 1])
            .repeated(to: [deps, rows, cols])
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //----------------------------------
    /// repeating a matrix `AnyConvertable`
    init<C>(repeating elements: C, deps: Int, name: String? = nil) where
        C: Collection, C.Element: Collection,
        C.Element.Element: AnyConvertable, Element: AnyConvertable
    {
        let rows = elements.count
        let cols = elements[elements.startIndex].count
        let shape = DataShape(extents: [1, rows, cols])
            .repeated(to: [deps, rows, cols])
        self = Self.create(elements.joined().lazy.map { Element(any: $0) },
                           shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(referenceTo buffer: UnsafeBufferPointer<Element>,
         deps: Int, rows: Int, cols: Int,
         name: String? = nil)
    {
        let shape = DataShape(extents: [deps, rows, cols])
        self = Self.create(referenceTo: buffer, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read write buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
         deps: Int, rows: Int, cols: Int,
         name: String? = nil)
    {
        let shape = DataShape(extents: [deps, rows, cols])
        self = Self.create(referenceTo: buffer, shape, name)
    }
    
    
    //--------------------------------------------------------------------------
    // typed views
    func createBoolTensor(with extents: [Int]) -> Volume<Bool> {
        Volume<Bool>(extents: extents)
    }
    
    func createIndexTensor(with extents: [Int]) -> Volume<IndexElement> {
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
        let dRange = makePositive(range: d, count: extents[0])
        let rRange = makePositive(range: r, count: extents[1])
        let cRange = makePositive(range: c, count: extents[2])
        let viewPosition = [dRange.lowerBound,
                            rRange.lowerBound,
                            cRange.lowerBound]
        let viewExtents = [dRange.count, rRange.count, cRange.count]
        return view(at: viewPosition, extents: viewExtents)
    }
    
    @inlinable @inline(__always)
    subscript(_ d: RangeInterval, r: RangeInterval, c: RangeInterval) -> Self {
        let dRange = makePositive(range: d, count: extents[0])
        let rRange = makePositive(range: r, count: extents[1])
        let cRange = makePositive(range: c, count: extents[2])
        let viewPosition = [dRange.from, rRange.from, cRange.from]
        let viewExtents = [dRange.to, rRange.to, cRange.to]
        let steps = [dRange.step, rRange.step, cRange.step]
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
    public let shape: DataShape
    public var tensorArray: TensorArray<Element>
    public let viewOffset: Int
    public let singleElementExtents = [1, 1, 1]
    
    public init(shape: DataShape,
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

