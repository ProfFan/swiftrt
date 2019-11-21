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
// >>>>>> User API <<<<<<
///
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

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    func concat<T>(tensors: [T], alongAxis axis: Int, result: inout T) where
        T: TensorView
    {
        do {
            // Note: if the tensors are large then they could be copied in parallel
            let shared = try result.sharedView(using: self)
            var index = [Int](repeating: 0, count: tensors[0].rank)
            
            for tensor in tensors {
                var outView = shared.view(at: index, extents: tensor.extents)
                tensor.map(into: &outView) { $0 }
                index[axis] += tensor.extents[axis]
            }
        } catch {
            DeviceContext.report(error)
        }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func concat<T>(tensors: [T], alongAxis axis: Int, result: inout T) where
        T: TensorView
    {
        let inputs: () throws -> ([TensorValueCollection<T>]) = {
            tensors.map { $0.elements(using: self) }
        }

        let outputs: () throws -> ([TensorMutableValueCollection<T>]) = {
            var index = [Int](repeating: 0, count: tensors[0].rank)
            let shared = try result.sharedView(using: self)
            var outCollections = [TensorMutableValueCollection<T>]()
            
            for tensor in tensors {
                var view = shared.view(at: index, extents: tensor.extents)
                outCollections.append(view.mutableElements(using: self))
                index[axis] += tensor.extents[axis]
            }
            return outCollections
        }

        queue(#function, inputs, outputs) { inSeqs, outSeqs in
            for i in 0..<inSeqs.count {
                for (j, k) in zip(inSeqs[i].indices, outSeqs[i].indices) {
                    outSeqs[i][k] = inSeqs[i][j]
                }
            }
        }
    }
}
#endif

//==============================================================================
// >>>>>> User API <<<<<<
/// fill<T>(result:value:
/// fills the view with the specified value
public func fill<T>(_ result: inout T, with value: T.Element) where
    T: TensorView
{
    DeviceContext.currentQueue.fill(&result, with: value)
}

/// fillWithIndex(x:startAt:
/// fills the view with the spatial sequential index
public func fillWithIndex<T>(_ result: inout T, startAt index: Int = 0) where
    T: TensorView, T.Element: AnyNumeric
{
    DeviceContext.currentQueue.fillWithIndex(&result, startAt: index)
}

public extension TensorView where Element: AnyNumeric {
    func filledWithIndex(startAt index: Int = 0) -> Self {
        var result = createDense()
        SwiftRT.fillWithIndex(&result, startAt: index)
        return result
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    /// fills the view with the scalar value
    func fill<T>(_ result: inout T, with value: T.Element) where T: TensorView {
        // TODO: can we hide the values/mutable values collections
        var values = result.mutableElements()
        for index in values.indices {
            values[index] = value
        }
    }
    
    /// fills the view with the spatial sequential index
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T: TensorView, T.Element: AnyNumeric
    {
        // TODO: can we hide the values/mutable values collections
        var value = startAt
        var values = result.mutableElements()
        for index in values.indices {
            values[index] = T.Element(any: value)
            value += 1
        }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    //--------------------------------------------------------------------------
    /// fill(result:with:
    /// NOTE: this can be much faster, doesn't need to be ordered access
    func fill<T>(_ result: inout T, with value: T.Element) where T: TensorView {
        queue(#function, {}, &result) {
            //            try result.readWrite().initialize(repeating: value)
            for index in $1.indices { $1[index] = value }
        }
    }
    
    //--------------------------------------------------------------------------
    /// fillWithIndex(x:startAt:
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T: TensorView, T.Element: AnyNumeric
    {
        queue(#function, {}, &result) {
            var value = startAt
            for index in $1.indices {
                $1[index] = T.Element(any: value)
                value += 1
            }
        }
    }
}
#endif
