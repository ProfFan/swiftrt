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
// casting for convertable types
public extension TensorView where Element: AnyConvertable {
    //--------------------------------------------------------------------------
    /// casting
    init<U>(_ other: U) where U: TensorView, U.Element: AnyConvertable {
        self = cast(other)
    }
}

//==============================================================================
//
public extension TensorView {
    //--------------------------------------------------------------------------
    // the fully specified initializers are implemented on concrete types

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



    
    
    init(element type: Element.Type) {
        self.init(shape: DataShape(),
                  tensorArray: TensorArray(),
                  viewOffset: 0,
                  isShared: false)
    }

    //--------------------------------------------------------------------------
    /// creates a tensor of the same type and shape as `self` with `Element`
    /// equal to `Bool`
    func createBoolTensor() -> BoolView {
        return createBoolTensor(with: extents)
    }
    
    /// creates a tensor of the same type and shape as `self`
    
    /// creates a tensor of the same shape as `self` with `Element`
    /// equal to `IndexElement`
    func createIndexTensor() -> IndexView {
        return createIndexTensor(with: extents)
    }
    
    //--------------------------------------------------------------------------
    /// repeated view
    init(repeating other: Self, extents: [Int]) {
        // make sure other has valid extents
        assert({
            for i in 0..<other.rank {
                if other.extents[i] != 1 && other.extents[i] != extents[i] {
                    return false
                }
            }
            return true
        }(), "repeated tensor extents must be either 1" +
            " or match the new tensor extents")
        
        // compute strides, setting stride to 0 for repeated dimensions
        var strides = [Int](repeating: 0, count: extents.count)
        for i in 0..<other.rank where other.extents[i] == extents[i] {
            strides[i] = other.shape.strides[i]
        }
        
        self.init(shape: DataShape(extents: extents, strides: strides),
                  tensorArray: other.tensorArray,
                  viewOffset: other.viewOffset,
                  isShared: other.isShared)
    }
    
    //--------------------------------------------------------------------------
    /// concatenated tensors
    init(concatenating others: Self...,
        along axis: Int = 0,
        name: String? = nil)
    {
        self = Self(concatenating: others, along: axis, name: name)
    }
    
    init(concatenating tensors: [Self],
         along axis: Int = 0,
         name: String? = nil)
    {
        self = SwiftRT.concat(tensors: tensors, along: axis, name: name)
    }
    
    //--------------------------------------------------------------------------
    /// createDense(shape:
    func createDense(with shape: DataShape, name: String? = nil) -> Self {
        let name = name ?? String(describing: Self.self)
        let array = TensorArray<Element>(count: shape.elementCount, name: name)
        return Self(shape: shape.dense,
                    tensorArray: array,
                    viewOffset: 0,
                    isShared: false)
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
    func createDense() -> Self { return createDense(with: self.shape) }
    
    //--------------------------------------------------------------------------
    /// createSingleElement
    /// helper to create a rank extended value
    func createSingleElement(name: String? = nil) -> Self {
        let name = name ?? String(describing: Self.self)
        let shape = DataShape(extents: singleElementExtents,
                              strides: singleElementExtents)
        let array = TensorArray<Element>(count: 1, name: name)
        return Self(shape: shape, tensorArray: array, viewOffset: 0,
                    isShared: false)
    }
    
    //--------------------------------------------------------------------------
    /// create(repeating:
    func create(repeating value: Element, name: String? = nil) -> Self {
        let name = name ?? String(describing: Self.self)
        let strides = [Int](repeating: 0, count: rank)
        let shape = DataShape(extents: extents, strides: strides)
        let array = TensorArray<Element>(count: 1, name: name)
        var view = Self(shape: shape, tensorArray: array, viewOffset: 0,
                        isShared: false)
        // we know we can get the cpu buffer
        try! view.readWrite()[0] = value
        return view
    }
    
}

