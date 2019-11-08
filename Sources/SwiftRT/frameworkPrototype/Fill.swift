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
/// - Parameter others: array of tensors whose elements will be joined
/// - Parameter axis: dimension to append the elements
func concat<T>(others: [T], along axis: Int = 0, result: inout T) where
    T: TensorView
{
    DeviceContext.currentQueue.concat(others: others, along: axis,
                                      result: &result)
}

public extension TensorView {
    func concat(_ others: [Self], along axis: Int = 0) -> Self {
        var result = createDense()
        DeviceContext.currentQueue.concat(others: others, along: axis,
                                          result: &result)
        return result
    }

    func concat(_ others: Self..., along axis: Int = 0) -> Self {
        return concat(others, along: axis)
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceFunctions {
    func concat<T>(others: [T], along axis: Int, result: inout T) where
        T: TensorView
    {
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceFunctions])
// target generated from Intent by the compiler
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func concat<T>(others: [T], along axis: Int, result: inout T) where
        T: TensorView
    {
//        queue(#function, {}, &result) {
//        }
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
        DeviceContext.currentQueue.fillWithIndex(&result, startAt: index)
        return result
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceFunctions {
    /// fills the view with the scalar value
    func fill<T>(_ result: inout T, with value: T.Element) where T: TensorView {
        // TODO: can we hide the values/mutable values collections
        var values = result.mutableValues()
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
        var values = result.mutableValues()
        for index in values.indices {
            values[index] = T.Element(any: value)
            value += 1
        }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceFunctions])
// target generated from Intent by the compiler
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
