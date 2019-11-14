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
// tensor subscripting helpers

/// makePositive(range:count:
/// - Parameter range: a range expression specifying the bounds of
/// the desired range. Negative bounds are relative to the end of the range
/// - Parameter count: the number of elements in the collection that
/// the range calculation should be relative to.
/// - Returns: a positive range relative to the specified bounding `count`
@inlinable @inline(__always)
public func makePositive<R>(range: R, count: Int) -> Range<Int> where
    R: RangeExpression, R.Bound == Int
{
    let count = count - 1
    let r = range.relative(to: -count..<count + 1)
    let lower = r.lowerBound < 0 ? r.lowerBound + count : r.lowerBound
    let upper = r.upperBound < 0 ? r.upperBound + count : r.upperBound
    return lower..<upper
}

/// makeStepped(view:parent:steps:
/// computes the extents and strides for creating a stepped subview
/// - Parameter view: the extents of the desired view in parent coordinates
/// - Parameter parent: the strides of the parent view
/// - Parameter steps: the step interval along each dimension
/// - Returns: the extents and strides to be used to create a subview
@inlinable @inline(__always)
public func makeStepped(view extents: [Int],
                        parent strides: [Int],
                        steps: [Int]) -> (extents: [Int], strides: [Int])
{
    assert(extents.count == strides.count && extents.count == steps.count)
    let subExtents = zip(extents, steps).map {
        $0 / $1 + ($0 % $1 == 0 ? 0 : 1)
    }
    let subStrides = zip(strides, steps).map { $0 * $1 }
    return (subExtents, subStrides)
}

//==============================================================================
// Codable extensions
extension Matrix: Codable where Element: Codable {}

//==============================================================================
// MatrixView
public protocol MatrixView: TensorView {}

extension Matrix: CustomStringConvertible where Element: AnyConvertable {
    public var description: String { return formatted() }
}

//==============================================================================
// MatrixView extensions
public extension MatrixView {
    var startIndex: MatrixIndex { return MatrixIndex(view: self, at: (0, 0)) }
    var endIndex: MatrixIndex { return MatrixIndex(endOf: self) }

    //--------------------------------------------------------------------------
    // cast
    init<U>(_ other: U) where
        Self.Element: AnyConvertable,
        U: MatrixView, U.Element: AnyConvertable
    {
        fatalError()
    }

    //--------------------------------------------------------------------------
    /// empty array
    init(_ extents: MatrixExtents, name: String? = nil) {
        let shape = DataShape(extents: [extents.rows, extents.cols])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(count: shape.elementCount, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// repeating
    init(_ extents: MatrixExtents, repeating other: Self) {
        let extents = [extents.rows, extents.cols]
        self.init(with: extents, repeating: other)
    }
    
    //-------------------------------------
    /// with reference to read only buffer
    /// useful for memory mapped databases, or hardware device buffers
    init(_ extents: MatrixExtents,
         layout: MatrixLayout = .rowMajor,
         referenceTo buffer: UnsafeBufferPointer<Element>,
         name: String? = nil)
    {
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(referenceTo: buffer, name: name)

        // create shape considering column major
        let extents = [extents.rows, extents.cols]
        let shape = layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
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

    //-------------------------------------
    /// with single value
    init(_ element: Element, name: String? = nil) {
        let shape = DataShape(extents: [1, 1])
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: [element], name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
    
    //-------------------------------------
    /// with convertible collection
    init<C>(_ extents: MatrixExtents, name: String? = nil,
            layout: MatrixLayout = .rowMajor, with any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init(extents, name: name, layout: layout,
                  elements: any.lazy.map { Element(any: $0) })
    }

    //-------------------------------------
    /// with convertible collection
    init<C>(_ rows: Int, _ cols: Int, name: String? = nil,
            layout: MatrixLayout = .rowMajor, with any: C) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self.init((rows, cols), name: name, layout: layout,
                  elements: any.lazy.map { Element(any: $0) })
    }

    //-------------------------------------
    /// with an element collection
    init<C>(_ extents: MatrixExtents, name: String? = nil,
            layout: MatrixLayout = .rowMajor, elements: C) where
        C: Collection, C.Element == Element
    {
        let extents = [extents.rows, extents.cols]
        let shape = layout == .rowMajor ?
            DataShape(extents: extents) :
            DataShape(extents: extents).columnMajor()
        assert(shape.elementCount == elements.count, countMismatch)
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: elements, name: name)
        self.init(shape: shape, tensorArray: array,
                  viewOffset: 0, isShared: false)
    }
}

//==============================================================================
// range subscripting
public extension MatrixView {
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

