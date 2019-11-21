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

// hack for now to bridge getting to Swift auto differentiation
#if !canImport(TensorFlow)
public protocol Differentiable {}
#endif

//==============================================================================
/// TensorView protocol
/// A TensorView object is the primary interface for working with data in
/// the app and on various devices. Specialized shaped instances such as
/// Vector, Matrix, Volume, etc.. adopt this protocol. They will generically
/// be referred to as tensors after this.
///
/// Data can be safely accessed on the app thread and asynchronously on
/// device queues without the user needing be concerned with synchronization.
///
/// When a tensor is created, no memory is allocated until the first time
/// access is requested. The location of the access determines where the
/// buffer is created. No host shadow buffer is created. So temporary tensors
/// on local discrete devices or remote hosts can be freely created and
/// manipulated without any host resources being used, or data being transited
/// to the target device.
///
/// Data replication and synchronization are transparent to the user.
///
/// TensorViews are references to data and respect copy on write semantics,
/// locally and on device. Many operations can be performed with zero copy.
///
/// Data repeating (broadcasting) is an instrinsic feature
///
public protocol TensorView: Logging {
    //--------------------------------------------------------------------------
    /// the type of element stored by the tensor
    associatedtype Element
    /// A tensor shape specific indexer used to calculate a data buffer
    /// index based on a view's spatial position
    associatedtype Index: TensorIndexing
    /// the type of read only elements collection
    associatedtype Values: RandomAccessCollection
        where Values.Element == Element
    /// the type of read write elements collection
    associatedtype MutableValues: RandomAccessCollection & MutableCollection
        where MutableValues.Element == Element
    /// A concrete type used in generics to pass Boolean values
    associatedtype BoolView: TensorView where BoolView.Element == Bool
    /// A concrete type used in generics to return index results
    associatedtype IndexView: TensorView where IndexView.Element == IndexElement

    //--------------------------------------------------------------------------
    // properties
    /// returns an index one past the end of the tensor used for collections
    var endIndex: Index { get }
    /// used internally when obtaining write access to manage
    /// multi-threaded writes without causing `tensorArray` copy on write.
    var isShared: Bool { get }
    /// format describes how to interpret the meaning of each dimension
    var format: TensorFormat { get }
    /// the shape of the view used for indexing
    var shape: DataShape { get }
    /// returns the first tensor index used for collections
    var startIndex: Index { get }
    /// class reference to the underlying byte buffer
    var tensorArray: TensorArray<Element> { get set }
    /// the linear element offset where the view begins
    var viewOffset: Int { get }
    /// a static array of 1's with count == rank
    /// used for creating reduction result tensors
    var singleElementExtents: [Int] { get }
    
    //--------------------------------------------------------------------------
    /// fully specified used for creating views
    init(shape: DataShape,
         tensorArray: TensorArray<Element>,
         viewOffset: Int,
         isShared: Bool)

    //--------------------------------------------------------------------------
    /// creates a new dense tensor of the same type with the specified extents
    func createDense(with extents: [Int], name: String?) -> Self
    /// creates a new dense tensor of the same extent with the specified value
    func create(repeating: Element, name: String?) -> Self
    /// creates a new dense tensor where `Element` equals `Bool`
    /// with the specified extents
    func createBoolTensor(with extents: [Int]) -> BoolView
    /// creates a new dense tensor where `Element` equals `IndexElement`
    /// with the specified extents and initial values
    func createIndexTensor(with extents: [Int]) -> IndexView

    //--------------------------------------------------------------------------
    /// returns a collection of viewed elements
    func elements(using queue: DeviceQueue?) -> Values

    /// returns a collection of mutable viewed elements
    mutating func mutableElements(using queue: DeviceQueue?) -> MutableValues
}

//==============================================================================
//
public extension TensorView {
    //--------------------------------------------------------------------------
    /// returns a collection of read only elements
    func elements(using queue: DeviceQueue? = nil)
        -> TensorValueCollection<Self>
    {
        do {
            let buffer = try readOnly(using: queue)
            return TensorValueCollection(view: self, buffer: buffer)
        } catch {
            DeviceContext.report(error)
            return TensorValueCollection(view: self)
        }
    }
    
    var elements: TensorValueCollection<Self> { elements() }

    //--------------------------------------------------------------------------
    /// returns a collection of read write values
    mutating func mutableElements(using queue: DeviceQueue? = nil)
        -> TensorMutableValueCollection<Self>
    {
        do {
            let buffer = try readWrite(using: queue)
            return TensorMutableValueCollection(view: &self, buffer: buffer)
        } catch {
            DeviceContext.report(error)
            return TensorMutableValueCollection(view: &self)
        }
    }
}

//==============================================================================
/// DifferentiableTensorView
///
/// Marker protocol for `TensorView` that conform to `Differentiable`.
///
/// While this protoocl is not strictly necessary, it is used to reduce the
/// number of generic requirements when writing `@differentiable` attributes on
/// generic differentiable `TensorView` functions.
public protocol DifferentiableTensorView: TensorView & Differentiable where
    Self == TangentVector, Element: AnyDifferentiableScalar {}

//==============================================================================
// view subscripting helpers

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

/// makePositive(range:count:
/// - Parameter range: a range expression specifying the bounds of
/// the desired range. Negative bounds are relative to the end of the range
/// - Parameter count: the number of elements in the collection that
/// the range calculation should be relative to.
/// - Returns: a positive range relative to the specified bounding `count`
@inlinable @inline(__always)
public func makePositive(range: RangeInterval, count: Int)
    -> ResolvedRangeInterval
{
    var from = range.from ?? 0
    if from < 0 { from += count }
    var to = range.to ?? count
    if to < 0 { to += count }
    return (from, to, range.step ?? 1)
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
/// IndexElement
/// The data type used for tensors that contain tensor spatial index values
public typealias IndexElement = Int32

//==============================================================================
/// ScalarType
/// Used primarily for serialization, C APIs, and Cuda kernels
// TODO: maybe remove this after Cuda integration if not used
public enum ScalarType: Int {
    // integers
    case real8U, real8I, real16U, real16I, real32U, real32I, real64U, real64I
    // floats
    case real16F, real32F, real64F
    // non numeric
    case bool
}

//==============================================================================
/// TensorFormat
/// an enumeration describing how to interpret the meaning of each
/// dimension in a tensor.
///
/// n: the number of items in the set
/// d: the number of depths per item
/// h: the number of rows per depth
/// w: the number of columns in a row
/// c: the number of channels per column
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor-descriptor
public enum TensorFormat: Int, Codable {
    // simple 0-3D layouts
    case scalar, vector, matrix, volume
    /// 4D layouts
    case nchw, nhwc
    /// 5D layouts
    case ncdhw, ndhwc
}

//==============================================================================
// TensorView default implementation
public extension TensorView where Element: AnyElement {
    /// element
    /// - Returns: the first element in the tensor
    var first: Element {
        do {
            return try readOnly()[0]
        } catch {
            DeviceContext.report(error)
            return Element()
        }
    }

    /// - Returns: the only element in the tensor
    var element: Element {
        assert(shape.elementCount == 1, "the `element` property expects " +
            "the tensor to have a single Element. Use `first` for sets")
        return first
    }
}

//==============================================================================
// TensorView default implementation
public extension TensorView {
    //--------------------------------------------------------------------------
    /// the number of elements in the collection
    var elementCount: Int { return shape.elementCount }
    /// the extents of the view
    var extents: [Int] { return shape.extents }
    /// `true` if the values are contiguosly arranged in memory
    var isContiguous: Bool { return shape.isContiguous }
    /// the number of items in the tensor, which is equal to `extents[0]`
    var items: Int { return shape.extents[0] }
    /// is `true` if the last data access caused the view's underlying
    /// tensorArray object to be copied.
    /// Used primarily for debugging and unit testing
    var lastAccessMutatedView: Bool { return tensorArray.lastAccessMutatedView }
    /// the name of the view, which can optionally be set to aid in debugging
    var name: String { return tensorArray.name }
    /// the number of dimensions in the view
    var rank: Int { return shape.rank }
    //--------------------------------------------------------------------------
    /// createView
    /// Returns a view of the tensorArray relative to this view
    private func createView(at offset: [Int], extents: [Int],
                            strides: [Int], isReference: Bool) -> Self {
        // validate
        assert(offset.count == shape.rank && extents.count == shape.rank)
        assert(shape.contains(offset: offset, extents: extents))

        // the subview offset is the current plus the offset of index
        let dataOffset = viewOffset + shape.linearIndex(of: offset)
        return Self(shape: DataShape(extents: extents, strides: strides),
                    tensorArray: tensorArray,
                    viewOffset: dataOffset,
                    isShared: isReference)
    }
    
    //--------------------------------------------------------------------------
    /// sharedView
    /// creation of a sharedView is for the purpose of reshaped writes
    /// and host multi-threaded writes to prevent mutation.
    /// The data will be copied before view creation if
    /// not uniquely held. Shared views will not perform
    /// copy-on-write when a write pointer is taken
    mutating func sharedView(using queue: DeviceQueue) throws -> Self {
        // get the queue, if we reference it as a tensorArray member it
        // it adds a ref count which messes things up
        let accessQueue = tensorArray.accessQueue
        
        return try accessQueue.sync {
            try copyIfMutates(using: queue)
            return Self(shape: shape,
                        tensorArray: tensorArray,
                        viewOffset: viewOffset,
                        isShared: true)
        }
    }

    //--------------------------------------------------------------------------
    /// repeated(to extents:
    func repeated(to extents: [Int]) -> Self {
        return Self(shape: shape.repeated(to: extents),
                    tensorArray: tensorArray,
                    viewOffset: viewOffset,
                    isShared: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// flattened
    /// Returns a view with all dimensions higher than `axis` set to 1
    /// and the extent of `axis` adjusted to be the new total element count
    func flattened(axis: Int = 0) -> Self {
        // check if self already meets requirements
        guard self.isShared != isShared || axis != shape.rank - 1 else {
            return self
        }
        
        // create flattened view
        return Self(shape: shape.flattened(),
                    tensorArray: tensorArray,
                    viewOffset: viewOffset,
                    isShared: isShared)
    }

    //--------------------------------------------------------------------------
    /// realized
    /// create a dense view where the elements are coalesced
    /// if it is already of the correct form, then `self` is reaturned
    func realized() throws -> Self {
        if shape.isContiguous {
            return self
        } else {
            var result = createDense()
            DeviceContext.currentQueue.copy(from: self, to: &result)
            return result
        }
    }
    
    //--------------------------------------------------------------------------
    /// an array of viewed elements
    @inlinable @inline(__always)
    var array: [Element] {
        return [Element](elements())
    }

    //--------------------------------------------------------------------------
    /// an array of viewed elements
    @inlinable @inline(__always)
    var flatArray: [Element] {
        return [Element](elements())
    }
    
    //--------------------------------------------------------------------------
    /// get a single value at the specified index
    @inlinable @inline(__always)
    func value(at position: Index.Position) throws -> Element {
        let buffer = try readOnly()
        let index = Index(view: self, at: position)
        return buffer[index.dataIndex]
    }
    
    //--------------------------------------------------------------------------
    /// set a single value at the specified index
    @inlinable @inline(__always)
    mutating func set(value: Element, at position: Index.Position) throws {
        let buffer = try readWrite()
        let index = Index(view: self, at: position)
        buffer[index.dataIndex] = value
    }
    
//    //--------------------------------------------------------------------------
//    /// squeezed(axes:
//    /// performs a rank reduction by removing dimensions with an extent of 1
//    /// - Parameter axes: the axes to squeeze. `nil` implies all axes.
//    /// - Returns: the new data shape
//    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`
//    func squeezed(axes: [Int]? = nil) -> NDTensor<Element> {
//        return NDTensor<Element>(shape: shape.squeezed(axes: axes),
//                                 tensorArray: tensorArray,
//                                 viewOffset: viewOffset,
//                                 isShared: isShared)
//    }

    //--------------------------------------------------------------------------
    /// isUniqueReference
    /// `true` if this view is the only view holding a reference to tensorArray
    mutating func isUniqueReference() -> Bool {
        return isKnownUniquelyReferenced(&tensorArray)
    }
    
    //--------------------------------------------------------------------------
    /// copyIfMutates
    /// Creates a copy of the tensorArray if read-write access causes mutation
    /// NOTE: this must be called from inside the accessQueue.sync block
    mutating func copyIfMutates(using queue: DeviceQueue) throws {
        // for unit tests
        tensorArray.lastAccessMutatedView = false
        guard !isShared && !isUniqueReference() else { return }
        
        diagnostic("\(mutationString) \(name)(\(tensorArray.trackingId)) " +
            "\(String(describing: Element.self))[\(shape.elementCount)]",
            categories: [.dataCopy, .dataMutation])
        
        // create the new array and copy the values
        tensorArray = try TensorArray<Element>(copying: tensorArray,
                                               using: queue)
        tensorArray.lastAccessMutatedView = true
    }

    //--------------------------------------------------------------------------
    /// synchronizeQueues
    /// If the queue is changing, then this creates an event and
    /// records it onto the end of the lastQueue, then records a wait
    /// on the new queue. This insures the lastQueue finishes before
    /// the new one begins
    private func synchronize(queue lastQueue: DeviceQueue?,
                             with nextQueue: DeviceQueue) throws {
        if let lastQueue = lastQueue, nextQueue !== lastQueue {
            let event = try lastQueue.createEvent()
            diagnostic(
                "\(nextQueue.device.name)_\(nextQueue.name) will wait for " +
                "\(lastQueue.device.name)_\(lastQueue.name) " +
                "using QueueEvent(\(event.trackingId))",
                categories: .queueSync)
            try nextQueue.wait(for: lastQueue.record(event: event))
        }
    }
    
    //--------------------------------------------------------------------------
    /// readOnly(using queue:
    /// Returns a read only device memory buffer synced with the specified
    /// queue.
    func readOnly(using queue: DeviceQueue? = nil) throws
        -> UnsafeBufferPointer<Element>
    {
        // if no queue is specified then use the hostQueue
        let deviceQueue = queue ?? DeviceContext.hostQueue
        if let lastError = deviceQueue.lastError { throw lastError }

        // get the queue, if we reference it directly as a dataArray member it
        // it adds a ref count which messes things up
        let accessQueue = tensorArray.accessQueue
        
        return try accessQueue.sync {
            // this is only used for unit testing
            tensorArray.lastAccessMutatedView = false

            // sync queues
            try synchronize(queue: tensorArray.lastMutatingQueue,
                            with: deviceQueue)
            // get the buffer
            let buffer = try tensorArray.readOnly(using: deviceQueue)

            // if `queue` is nil then the deviceQueue is the hostQueue
            // and the caller wants to synchronize with the app thread
            if queue == nil {
                assert(deviceQueue.device.memory.addressing == .unified)
                try deviceQueue.waitUntilQueueIsComplete()
            }

            return UnsafeBufferPointer(
                start: buffer.baseAddress!.advanced(by: viewOffset),
                count: shape.elementSpanCount)
        }
    }
    
    //--------------------------------------------------------------------------
    /// deviceReadOnly(using queue:
    /// Returns a read only device raw memory pointer synced with the specified
    /// queue.
    func deviceReadOnly(using queue: DeviceQueue? = nil) throws
        -> UnsafeRawPointer
    {
        return try UnsafeRawPointer(readOnly(using: queue).baseAddress!)
    }
    
    //--------------------------------------------------------------------------
    /// readWrite(using queue:
    /// Returns a read write device memory buffer synced with the specified
    /// queue.
    mutating func readWrite(using queue: DeviceQueue? = nil) throws
        -> UnsafeMutableBufferPointer<Element>
    {
        precondition(!tensorArray.isReadOnly, "the tensor is read only")
        let deviceQueue = queue ?? DeviceContext.hostQueue
        if let lastError = deviceQueue.lastError { throw lastError }

        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let accessQueue = tensorArray.accessQueue
        
        return try accessQueue.sync {
            // sync queues
            try synchronize(queue: tensorArray.lastMutatingQueue,
                            with: deviceQueue)
            // mutating write?
            try copyIfMutates(using: deviceQueue)

            // get the buffer
            let buffer = try tensorArray.readWrite(using: deviceQueue)
            
            // if `queue` is nil then the deviceQueue is the hostQueue
            // and the caller wants to synchronize with the app thread
            if queue == nil {
                assert(deviceQueue.device.memory.addressing == .unified)
                try deviceQueue.waitUntilQueueIsComplete()
            }

            return UnsafeMutableBufferPointer(
                start: buffer.baseAddress!.advanced(by: viewOffset),
                count: shape.elementSpanCount)
        }
    }
    
    //--------------------------------------------------------------------------
    /// deviceReadWrite(using queue:
    /// Returns a read write device raw memory pointer synced with the specified
    /// queue.
    mutating func deviceReadWrite(using queue: DeviceQueue? = nil) throws
        -> UnsafeMutableRawPointer
    {
        return try UnsafeMutableRawPointer(readWrite(using: queue).baseAddress!)
    }
    
    //--------------------------------------------------------------------------
    /// view
    /// Create a sub view of the tensorArray relative to this view
    func view(at offset: [Int], extents: [Int]) -> Self {
        // the view created will have the same isShared state as the parent
        return createView(at: offset, extents: extents,
                          strides: shape.strides, isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// view
    /// Create a sub view of the tensorArray relative to this view
    func view(at offset: [Int], extents: [Int], strides: [Int]) -> Self {
        // the view created will have the same isShared state as the parent
        return createView(at: offset, extents: extents,
                          strides: strides, isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// viewItems
    /// Returns a view along the first dimension spanning all the others.
    /// It is used to simplify accessing a set of training samples.
    /// The view created will have the same isShared state as the parent
    func viewItems(at offset: Int, count: Int) -> Self {
        let index, viewExtents: [Int]
        if rank == 1 {
            index = [offset]
            viewExtents = [count]
        } else {
            index = [offset] + [Int](repeating: 0, count: rank - 1)
            viewExtents = [count] + shape.extents.suffix(from: 1)
        }
        
        return createView(at: index, extents: viewExtents,
                          strides: shape.strides, isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// view(item:
    func view(item: Int) -> Self {
        return viewItems(at: item, count: 1)
    }
}

//==============================================================================
public extension TensorView {
    //--------------------------------------------------------------------------
    /// hostMultiWrite
    /// divides a tensor into mutable batches and concurrently passes them
    /// to the `body` for processing
    /// - Parameter batchSize: the number of items to process at a time. The
    /// default is the total divided by the number of active cores
    /// - Parameter synchronous: if `true` the batches will be executed
    /// synchronously to aid in debugging
    /// - Parameter body: the function to perform
    mutating func hostMultiWrite(
        batchSize: Int? = nil,
        synchronous: Bool = false,
        _ body: @escaping (_ view: Self) throws
        -> Void) throws
    {
        assert(batchSize == nil || batchSize! <= extents[0])
        let queue = DeviceContext.hostQueue
        let errorDevice = queue.device
        var shared = try sharedView(using: queue)
        let group = DispatchGroup()
        let batchQueue = DispatchQueue(label: "hostMultiWrite",
                                       attributes: .concurrent)
        let batchSize = batchSize ?? {
            let size = extents[0] / ProcessInfo.processInfo.activeProcessorCount
            return size == 0 ? extents[0] : size
        }()
        let remainder = extents[0] % batchSize
        
        // do the work
        func queueBatch(item: Int, count: Int) throws {
            let view = shared.viewItems(at: item, count: count)
            if synchronous {
                try body(view)
            } else {
                guard queue.lastError == nil else { throw queue.lastError! }
                batchQueue.async(group: group) {
                    do {
                        try body(view)
                    } catch {
                        errorDevice.report(error)
                    }
                }
            }
        }
        
        // ensure the data is local
        _ = try shared.readWrite(using: queue)
        
        // launch the batches
        let lastBatchIndex = extents[0] - remainder
        for i in stride(from: 0, to: lastBatchIndex, by: batchSize) {
            try queueBatch(item: i, count: batchSize)
        }
        
        // process remaining items
        if remainder > 0 {
            try queueBatch(item: lastBatchIndex, count: remainder)
        }
        group.wait()
    }
    
    //--------------------------------------------------------------------------
    /// getHostMultiWriteBuffer
    /// Returns a read write host memory buffer synced with the host app
    /// queue.
    mutating func getHostMultiWriteBuffer() ->
        UnsafeMutableBufferPointer<Element>
    {
        assert(tensorArray.lastMutatingQueue != nil,
               "readWrite(using: DeviceContext.hostQueue) must be called first")
        let lastQueue = tensorArray.lastMutatingQueue!
        assert(lastQueue.device.memory.addressing == .unified)
        // get the queue, if we reference it as a dataArray member it
        // it adds a ref count which messes things up
        let queue = tensorArray.accessQueue
        
        return queue.sync {
            // the buffer is already in host memory so it can't fail
            let buffer = try! tensorArray.readWrite(using: lastQueue)
            
            return UnsafeMutableBufferPointer(
                start: buffer.baseAddress!.advanced(by: viewOffset),
                count: shape.elementSpanCount)
        }
    }
}

//==============================================================================
//
public extension TensorView where Element: FloatingPoint {
    //--------------------------------------------------------------------------
    /// isFinite
    /// `true` if all elements are finite values. Primarily used for debugging
    func isFinite() throws -> Bool {
        let values = try readOnly()
        for value in values {
            if !value.isFinite {
                return false
            }
        }
        return true
    }
}
