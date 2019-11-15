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
// error messages
let countMismatch = "the number of initial elements must equal the tensor size"

//==============================================================================
// shaped positions and extents used for indexing and selection
public typealias NDPosition = [Int]
public typealias VectorPosition = Int
public typealias VectorExtents = Int
public typealias MatrixPosition = (r: Int, c: Int)
public typealias MatrixExtents = (rows: Int, cols: Int)
public typealias VolumePosition = (d: Int, r: Int, c: Int)
public typealias VolumeExtents = (depths: Int, rows: Int, cols: Int)
public typealias NCHWPosition = (i: Int, ch: Int, r: Int, c: Int)
public typealias NCHWExtents = (items: Int, channels: Int, rows: Int, cols: Int)
public typealias NHWCPosition = (i: Int, r: Int, c: Int, ch: Int)
public typealias NHWCExtents = (items: Int, rows: Int, cols: Int, channels: Int)

public enum MatrixLayout { case rowMajor, columnMajor }

//==============================================================================
// Codable extensions
extension Matrix: Codable where Element: Codable {}

//==============================================================================
// MatrixView
public protocol MatrixView: TensorView {}

extension Matrix: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

public extension TensorView {
    static func create(_ shape: DataShape, _ name: String?) -> Self {
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(count: shape.elementCount, name: name)
        return Self(shape: shape.dense, tensorArray: array,
                    viewOffset: 0, isShared: false)
    }
    
    static func create(referenceTo buffer: UnsafeBufferPointer<Element>,
                       _ shape: DataShape, _ name: String?) -> Self {
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(referenceTo: buffer, name: name)
        return Self(shape: shape.dense, tensorArray: array,
                    viewOffset: 0, isShared: false)
    }

    static func create(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
                       _ shape: DataShape, _ name: String?) -> Self {
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(referenceTo: buffer, name: name)
        return Self(shape: shape.dense, tensorArray: array,
                    viewOffset: 0, isShared: false)
    }
    
    static func create<C>(_ elements: C, _ shape: DataShape,
                          _ name: String?) -> Self where
        C: Collection, C.Element == Element
    {
        assert(shape.elementCount == elements.count, countMismatch)
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: elements, name: name)
        return Self(shape: shape.dense, tensorArray: array,
                    viewOffset: 0, isShared: false)
    }
}

//==============================================================================
// MatrixView extensions
public extension MatrixView {
    //--------------------------------------------------------------------------
    /// reserved space
    init(_ rows: Int, _ cols: Int,
         layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        self = Self.create(Self.matrixShape([rows, cols], layout), name)
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
    
    //--------------------------------------------------------------------------
    /// repeating other
    init(repeating other: Self, rows: Int, cols: Int) {
        self.init(repeating: other, extents: [rows, cols])
    }

    /// repeating `Element` collection
    init<C>(repeatingElements elements: C,
            rows: Int = 1, cols: Int = 1,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let shape = Self.matrixShape([rows, cols], layout)
        self = Self.create(elements, shape, name)
    }
    
    /// repeating `AnyConvertable` collection
    init<C>(repeating elements: C, rows: Int = 1, cols: Int = 1,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let shape = Self.matrixShape([rows, cols], layout)
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
    /// BoolView
    func createBoolTensor(with extents: [Int]) -> Matrix<Bool> {
        let shape = DataShape(extents: extents)
        let array = TensorArray<Bool>(count: shape.elementCount,
                                      name: String(describing: Self.self))
        return Matrix<Bool>(shape: shape, tensorArray: array,
                            viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// IndexView
    func createIndexTensor(with extents: [Int]) -> Matrix<IndexElement> {
        let shape = DataShape(extents: extents)
        let name = String(describing: Self.self)
        let array = TensorArray<IndexElement>(count: shape.elementCount,
                                              name: name)
        return Matrix<IndexElement>(shape: shape, tensorArray: array,
                                    viewOffset: 0, isShared: false)
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
    // utility
    private static func matrixShape(_ extents: [Int],
                                    _ layout: MatrixLayout) -> DataShape {
        return layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
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

