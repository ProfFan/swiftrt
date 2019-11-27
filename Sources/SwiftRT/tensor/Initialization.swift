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

//==============================================================================
// error messages
let _messageElementCountMismatch =
"the number of initial elements must equal the tensor size"

let _messageNewTensorsShouldBeDense = "new tensors should be dense"

//==============================================================================
// casting for convertable types
public extension TensorView where Element: AnyConvertable {
    //--------------------------------------------------------------------------
    /// casting
    init<U>(_ other: U) where U: TensorView, U.Element: AnyConvertable {
        self = cast(other)
    }
}

public typealias RangeInterval = (from: Int?, to: Int?, step: Int?)
public typealias ResolvedRangeInterval = (from: Int, to: Int, step: Int)

//==============================================================================
//
public extension TensorView {
    //--------------------------------------------------------------------------
    /// empty
    /// creates an empty tensor that can be used where a return
    /// value is needed in an error condition.
    init() {
        self.init(shape: DataShape(),
                  tensorArray: TensorArray(),
                  viewOffset: 0,
                  isShared: false)
    }

    //--------------------------------------------------------------------------
    /// creates a tensor of the same type and shape as `self` with `Element`
    /// equal to `Bool`
    func createBoolTensor() -> BoolView { createBoolTensor(with: extents) }
    /// creates a tensor of the same shape as `self` with `Element`
    /// equal to `IndexElement`
    func createIndexTensor() -> IndexView { createIndexTensor(with: extents) }
    
    //--------------------------------------------------------------------------
    /// concatenated tensors
    init(concatenating tensors: Self..., alongAxis axis: Int = 0,
         name: String? = nil)
    {
        self = Self(concatenating: tensors, alongAxis: axis, name: name)
    }
    
    init(concatenating tensors: [Self], alongAxis axis: Int = 0,
         name: String? = nil)
    {
        self = SwiftRT.concat(tensors: tensors, alongAxis: axis, name: name)
    }
    
    //--------------------------------------------------------------------------
    /// repeating element
//    @differentiable(vjp: _vjpInit where Self: DifferentiableTensorView)
    init(repeating value: Element, to extents: [Int], name: String? = nil) {
        let strides = [Int](repeating: 0, count: extents.count)
        let shape = DataShape(extents: extents, strides: strides)
        self = Self.create([value], shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// repeating element
//    @differentiable(where Self: DifferentiableTensorView)
    init<U>(repeating value: Element, like other: U, name: String? = nil)
        where U: TensorView
    {
        self = Self(repeating: value, to: other.extents)
    }
    
    //--------------------------------------------------------------------------
    /// createDense(shape:
    func createDense(with shape: DataShape, name: String? = nil) -> Self {
        Self.create(shape.dense, name)
    }
    
    //--------------------------------------------------------------------------
    /// createDense(extents:
    func createDense(with extents: [Int], name: String? = nil) -> Self {
        let newShape = isContiguous ?
            DataShape(extents: extents, strides: self.shape.strides) :
            DataShape(extents: extents)
        return createDense(with: newShape, name: name)
    }
    
    //--------------------------------------------------------------------------
    /// createDense()
    func createDense() -> Self { return createDense(with: shape) }
    
    //--------------------------------------------------------------------------
    /// createReductionResult
    /// creates a tensor of suitable form to recieve a reduction result.
    func createReductionResult(alongAxes axes: Set<Int>?) -> Self {
        guard let axes = axes else { return createSingleElement() }
        assert(axes.isSubset(of: 0..<rank), "axis is out of bounds")
        var resultExtents = extents
        axes.forEach { resultExtents[$0] = 1 }
        return Self.create(DataShape(extents: resultExtents), nil)
    }

    //--------------------------------------------------------------------------
    /// createSingleElement
    /// helper to create a rank extended value
    func createSingleElement(name: String? = nil) -> Self {
        let shape = DataShape(extents: singleElementExtents,
                              strides: singleElementExtents)
        return Self.create(shape, name)
    }
    
    //==========================================================================
    // utility functions for creating shaped types
    static func create(_ shape: DataShape, _ name: String?) -> Self {
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(count: shape.elementCount, name: name)
        return Self(shape: shape, tensorArray: array,
                    viewOffset: 0, isShared: false)
    }
    
    static func create(referenceTo buffer: UnsafeBufferPointer<Element>,
                       _ shape: DataShape, _ name: String?) -> Self {
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(referenceTo: buffer, name: name)
        return Self(shape: shape, tensorArray: array,
                    viewOffset: 0, isShared: false)
    }
    
    static func create(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
                       _ shape: DataShape, _ name: String?) -> Self {
        assert(shape.elementCount == buffer.count,
               "shape count does not match buffer count")
        // create tensor data reference to buffer
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(referenceTo: buffer, name: name)
        return Self(shape: shape, tensorArray: array,
                    viewOffset: 0, isShared: false)
    }
    
    static func create<C>(_ elements: C, _ shape: DataShape,
                          _ name: String?) -> Self where
        C: Collection, C.Element == Element
    {
        // it can be less if the elements are being repeated
        assert(elements.count <= shape.elementCount,
               _messageElementCountMismatch)
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(elements: elements, name: name)
        return Self(shape: shape, tensorArray: array,
                    viewOffset: 0, isShared: false)
    }
}

//==============================================================================
//

public extension TensorView where Self: DifferentiableTensorView {
    static func _vjpInit(repeating value: Element, to extents: [Int], name: String?) ->
        (value: Self, pullback: (Self) -> (Element))
    {
        fatalError()
    }
}

