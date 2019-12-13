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
/// concat
/// - Parameter tensors: array of tensors whose elements will be joined
/// - Parameter axis: dimension to append the elements
func concat<T>(tensors: [T], alongAxis axis: Int = 0,
               name: String? = nil) -> T where T: TensorView
{
    assert(tensors.count > 1)
    let joined = tensors[0].shape.joined(with: tensors[1...].map { $0.shape },
                                         alongAxis: axis)
    var result = tensors[0].createDense(with: joined, name: name)
    DeviceContext.currentQueue.concat(tensors: tensors, alongAxis: axis,
                                      result: &result)
    return result
}

public extension TensorView {
    func concat(_ others: Self..., alongAxis axis: Int = 0) -> Self {
        SwiftRT.concat(tensors: [self] + others, alongAxis: axis)
    }
}

//==============================================================================
/// copy
/// copies the elements from `view` to `result`

/// with placement
/// - Parameter from view: tensor to be copied
/// - Parameter to result: the tensor where the result will be written
@inlinable @inline(__always)
public func copy<T>(from view: T, to result: inout T) where T: TensorView
{
    DeviceContext.currentQueue.copy(from: view, to: &result)
}

//==============================================================================
/// fill<T>(result:value:
/// fills the view with the specified value
public func fill<T>(_ result: inout T, with element: T.Element)
    where T: TensorView
{
    DeviceContext.currentQueue.fill(result: &result, with: element)
}

public extension TensorView {
    /// filled
    /// creates a tensor and fills on device
    func filled(with element: Element) -> Self {
        var result = createDense()
        fill(&result, with: element)
        return result
    }
}

//==============================================================================
/// fillWithIndex(x:startAt:
/// fills the view with the spatial sequential index
public func fillWithIndex<T>(_ result: inout T, startAt index: Int = 0) where
    T: TensorView, T.Element: AnyNumeric
{
    DeviceContext.currentQueue.fillWithIndex(result: &result, startAt: index)
}

public extension TensorView where Element: AnyNumeric {
    func filledWithIndex(startAt index: Int = 0) -> Self {
        var result = createDense()
        SwiftRT.fillWithIndex(&result, startAt: index)
        return result
    }
}

//==============================================================================
/// replace<T>(x:with:result:
/// fills the view with the specified value
public func replace<T>(x: T, with y: T, where condition: T.BoolView,
                       result: inout T) where T: TensorView
{
    DeviceContext.currentQueue.replace(x: x, with: y, where: condition,
                                       result: &result)
}

public extension TensorView where Element: Comparable {
    func replacing(with y: Self, where condition: BoolView) -> Self
    {
        var result = createDense()
        SwiftRT.replace(x: self, with: y, where: condition, result: &result)
        return result
    }

    func replacing(with value: Element, where condition: BoolView) -> Self {
        replacing(with: Self(repeating: value, like: self), where: condition)
    }
}
