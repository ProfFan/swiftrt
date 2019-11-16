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

infix operator **  : MultiplicationPrecedence
infix operator .<  : ComparisonPrecedence
infix operator .<= : ComparisonPrecedence
infix operator .>= : ComparisonPrecedence
infix operator .>  : ComparisonPrecedence
infix operator .== : ComparisonPrecedence
infix operator .!= : ComparisonPrecedence
infix operator .=

//==============================================================================
/// Elementwise add tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable @inline(__always)
public func add<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: Numeric
{
    var result = lhs.createDense()
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    DeviceContext.currentQueue.add(lhs: lhs, rhs: rhs, result: &result)
    return result
}

public extension TensorView where Element: Numeric {
    @inlinable @inline(__always)
    static func + (lhs: Self, rhs: Self) -> Self { add(lhs, rhs) }

    @inlinable @inline(__always)
    static func += (lhs: inout Self, rhs: Element) { lhs = lhs + rhs }
    
    @inlinable @inline(__always)
    static func += (lhs: inout Self, rhs: Self) { lhs = lhs + rhs }

    @inlinable @inline(__always)
    static func +(lhs: Self, rhs: Element) -> Self {
        lhs + lhs.create(repeating: rhs)
    }

    @inlinable @inline(__always)
    static func +(lhs: Element, rhs: Self) -> Self {
        rhs.create(repeating: lhs) + rhs
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    func add<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    {
        zip(lhs, rhs).map(into: &result, +)
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func add<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    {
        queue(#function, {
            (lhs.elements(using: self),
             rhs.elements(using: self))
        }, &result) {
            zip($0.0, $0.1).map(into: &$1, +)
        }
    }
}
#endif

//==============================================================================
/// Elementwise subtract tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable @inline(__always)
public func subtract<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: Numeric
{
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createDense()
    DeviceContext.currentQueue.subtract(lhs: lhs, rhs: rhs, result: &result)
    return result
}

public extension TensorView where Element: Numeric {
    @inlinable @inline(__always)
    static func - (lhs: Self, rhs: Self) -> Self { subtract(lhs, rhs) }

    @inlinable @inline(__always)
    static func -= (lhs: inout Self, rhs: Element) { lhs = lhs - rhs }
    
    @inlinable @inline(__always)
    static func -= (lhs: inout Self, rhs: Self) { lhs = lhs - rhs }

    @inlinable @inline(__always)
    static func - (lhs: Self, rhs: Element) -> Self {
        lhs - lhs.create(repeating: rhs)
    }

    @inlinable @inline(__always)
    static func - (lhs: Element, rhs: Self) -> Self {
        rhs.create(repeating: lhs) - rhs
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    func subtract<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    {
        zip(lhs, rhs).map(into: &result, -)
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func subtract<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    {
        queue(#function, {
            (lhs.elements(using: self),
             rhs.elements(using: self))
        }, &result) {
            zip($0.0, $0.1).map(into: &$1, -)
        }
    }
}
#endif

//==============================================================================
/// Element wise multiply tensors with broadcasting

/// in place
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the size is smaller than `lhs` then
///   broadcasting will be performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func mul<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: Numeric
{
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createDense()
    DeviceContext.currentQueue.mul(lhs: lhs, rhs: rhs, result: &result)
    return result
}

public extension TensorView where Element: Numeric {
    @inlinable @inline(__always)
    static func * (lhs: Self, rhs: Self) -> Self { mul(lhs, rhs) }
    
    @inlinable @inline(__always)
    static func *= (lhs: inout Self, rhs: Element) { lhs = lhs * rhs }

    @inlinable @inline(__always)
    static func *= (lhs: inout Self, rhs: Self) { lhs = lhs * rhs }
    
    @inlinable @inline(__always)
    static func * (lhs: Self, rhs: Element) -> Self {
        lhs * lhs.create(repeating: rhs)
    }

    @inlinable @inline(__always)
    static func * (lhs: Element, rhs: Self) -> Self {
        rhs.create(repeating: lhs) * rhs
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    func mul<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    {
        zip(lhs, rhs).map(into: &result, *)
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func mul<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    {
        queue(#function, {
            (lhs.elements(using: self),
             rhs.elements(using: self))
        }, &result) {
            zip($0.0, $0.1).map(into: &$1, *)
        }
    }
}
#endif

//==============================================================================
/// Element wise divide
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func div<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
    var result = lhs.createDense()
    DeviceContext.currentQueue.div(lhs: lhs, rhs: rhs, result: &result)
    return result
}

public extension TensorView where Element: FloatingPoint {
    @inlinable @inline(__always)
    static func / (lhs: Self, rhs: Self) -> Self { div(lhs, rhs) }

    @inlinable @inline(__always)
    static func /= (lhs: inout Self, rhs: Element) { lhs = lhs / rhs }

    @inlinable @inline(__always)
    static func /= (lhs: inout Self, rhs: Self) { lhs = lhs / rhs }

    @inlinable @inline(__always)
    static func / (lhs: Self, rhs: Element) -> Self {
        lhs / lhs.create(repeating: rhs)
    }

    @inlinable @inline(__always)
    static func / (lhs: Element, rhs: Self) -> Self {
        rhs.create(repeating: lhs) / rhs
    }
}

//------------------------------------------------------------------------------
// >>>>>> INTENT <<<<<<
// User device function
public extension DeviceQueue {
    func div<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: FloatingPoint
    {
        zip(lhs, rhs).map(into: &result, /)
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
#if canImport(CpuAsync)
public extension CpuAsynchronousQueue {
    func div<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: FloatingPoint
    {
        queue(#function, {
            (lhs.elements(using: self),
             rhs.elements(using: self))
        }, &result) {
            zip($0.0, $0.1).map(into: &$1, /)
        }
    }
}
#endif

