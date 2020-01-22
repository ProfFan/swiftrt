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
public protocol TensorView: Codable, Logging {
    //--------------------------------------------------------------------------
    /// the type of element stored by the tensor
    associatedtype Element: TensorElementConformance
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
    associatedtype BoolView: TensorView where
        BoolView.Element == Bool, BoolView.Shape == Shape
    /// A concrete type used in generics to return index results
    associatedtype IndexView: TensorView where
        IndexView.Element == IndexType, IndexView.Shape == Shape
    /// tensor shape
    associatedtype Shape: ShapeProtocol

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
    var shape: Shape { get }
    /// returns the first tensor index used for collections
    var startIndex: Index { get }
    /// class reference to the underlying byte buffer
    var tensorArray: TensorArray<Element> { get set }
    /// the linear element offset where the view begins
    var viewOffset: Int { get }
    
    //--------------------------------------------------------------------------
    /// fully specified used for creating views
    init(shape: Shape,
         tensorArray: TensorArray<Element>,
         viewOffset: Int,
         isShared: Bool)

    //--------------------------------------------------------------------------
    /// creates a new dense tensor of the same type with the specified extents
    func createDense(with extents: Shape.Array, name: String?) -> Self
    /// creates a new dense tensor where `Element` equals `Bool`
    /// with the specified extents
    func createBoolTensor(with extents: Shape.Array) -> BoolView
    /// creates a new dense tensor where `Element` equals `IndexType`
    /// with the specified extents and initial values
    func createIndexTensor(with extents: Shape.Array) -> IndexView

    //--------------------------------------------------------------------------
    /// returns a collection of viewed elements
    func elements(using queue: DeviceQueue?) -> Values

    /// returns a collection of mutable viewed elements
    mutating func mutableElements(using queue: DeviceQueue?) -> MutableValues
}

public typealias TensorElementConformance = AnyElement & Codable & Equatable

//==============================================================================
//
public extension TensorView {
    //--------------------------------------------------------------------------
    /// returns a collection of read only elements
    @inlinable
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
    
    @inlinable
    var elements: TensorValueCollection<Self> { elements() }

    //--------------------------------------------------------------------------
    /// returns a collection of read write values
    @inlinable
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
    /// first
    /// - Returns: the first element in the tensor
    @inlinable
    var first: Element {
        do {
            return try readOnly()[0]
        } catch {
            DeviceContext.report(error)
            return Element()
        }
    }

    /// element
    /// can get and set the value of a single element tensor.
    /// - Returns: the only element in the tensor
    @inlinable
    var element: Element {
        get {
            assert(shape.isScalar, "the `element` property expects " +
                "the tensor to have a single Element. Use `first` for sets")
            return first
        }
    }
}

//==============================================================================
// TensorView default implementation
public extension TensorView {
    //--------------------------------------------------------------------------
    /// the number of elements in the collection
    @inlinable
    var count: Int { shape.count }
    /// the extents of the view
    @inlinable
    var extents: Shape.Array { shape.extents }
    /// `true` if the values are contiguosly arranged in memory
    @inlinable
    var isContiguous: Bool { shape.isContiguous }
    /// the number of items in the tensor, which is equal to `extents[0]`
    @inlinable
    var items: Int { shape.items }
    /// is `true` if the last data access caused the view's underlying
    /// tensorArray object to be copied.
    /// Used primarily for debugging and unit testing
    @inlinable
    var lastAccessMutatedView: Bool { tensorArray.lastAccessMutatedView }
    /// the name of the view, which can optionally be set to aid in debugging
    @inlinable
    var name: String { tensorArray.name }
    /// the number of dimensions in the view
    @inlinable
    var rank: Int { shape.rank }
    /// the strides of the tensor elements
    @inlinable
    var strides: Shape.Array { shape.strides }
    /// an array of viewed elements
    @inlinable
    var flatArray: [Element] { [Element](elements()) }

    //--------------------------------------------------------------------------
    /// createView
    /// Returns a view of the tensorArray relative to this view
    @inlinable
    func createView(at offset: Shape.Array, extents: Shape.Array,
                    strides: Shape.Array, isReference: Bool) -> Self
    {
        // validate
        assert(offset.count == shape.rank && extents.count == shape.rank)
        assert(shape.contains(offset: offset, extents: extents))

        // the subview offset is the current plus the offset of index
        let subViewOffset = viewOffset + shape.linearIndex(of: offset)
        return Self(shape: Shape(extents: extents, strides: strides),
                    tensorArray: tensorArray,
                    viewOffset: subViewOffset,
                    isShared: isReference)
    }
    
    //--------------------------------------------------------------------------
    /// sharedView
    /// creation of a sharedView is for the purpose of reshaped writes
    /// and host multi-threaded writes to prevent mutation.
    /// The data will be copied before view creation if
    /// not uniquely held. Shared views will not perform
    /// copy-on-write when a write pointer is taken
    @inlinable
    mutating func sharedView(using queue: DeviceQueue,
                             reshaped: Shape? = nil) throws -> Self {
        // get the queue, if we reference it as a tensorArray member it
        // it adds a ref count which messes things up
        let accessQueue = tensorArray.accessQueue
        
        return try accessQueue.sync {
            try copyIfMutates(using: queue)
            return Self(shape: reshaped ?? shape,
                        tensorArray: tensorArray,
                        viewOffset: viewOffset,
                        isShared: true)
        }
    }

    //--------------------------------------------------------------------------
    /// repeated(to extents:
    @inlinable
    func repeated(to extents: Shape.Array) -> Self {
        return Self(shape: shape.repeated(to: extents),
                    tensorArray: tensorArray,
                    viewOffset: viewOffset,
                    isShared: isShared)
    }
    
    @inlinable
    func repeated(to extents: Shape.Tuple) -> Self {
        repeated(to: Shape.Array(extents))
    }
    
    //--------------------------------------------------------------------------
    /// isUniqueReference
    /// `true` if this view is the only view holding a reference to tensorArray
    @inlinable
    mutating func isUniqueReference() -> Bool {
        isKnownUniquelyReferenced(&tensorArray)
    }
    
    //--------------------------------------------------------------------------
    /// copyIfMutates
    /// Creates a copy of the tensorArray if read-write access causes mutation
    /// NOTE: this must be called from inside the accessQueue.sync block
    @inlinable
    mutating func copyIfMutates(using queue: DeviceQueue) throws {
        // for unit tests
        tensorArray.lastAccessMutatedView = false
        guard !isShared && !isUniqueReference() else { return }
        
        diagnostic("\(mutationString) \(name)(\(tensorArray.trackingId)) " +
            "\(String(describing: Element.self))[\(shape.count)]",
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
    @inlinable
    func synchronize(queue lastQueue: DeviceQueue?,
                     with nextQueue: DeviceQueue) throws
    {
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
    @inlinable
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
                count: shape.spanCount)
        }
    }
    
    //--------------------------------------------------------------------------
    /// deviceReadOnly(using queue:
    /// Returns a read only device raw memory pointer synced with the specified
    /// queue.
    @inlinable
    func deviceReadOnly(using queue: DeviceQueue? = nil) throws
        -> UnsafeRawPointer
    {
        try UnsafeRawPointer(readOnly(using: queue).baseAddress!)
    }
    
    //--------------------------------------------------------------------------
    /// readWrite(using queue:
    /// Returns a read write device memory buffer synced with the specified
    /// queue.
    @inlinable
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
                count: shape.spanCount)
        }
    }
    
    //--------------------------------------------------------------------------
    /// deviceReadWrite(using queue:
    /// Returns a read write device raw memory pointer synced with the specified
    /// queue.
    @inlinable
    mutating func deviceReadWrite(using queue: DeviceQueue? = nil) throws
        -> UnsafeMutableRawPointer
    {
        try UnsafeMutableRawPointer(readWrite(using: queue).baseAddress!)
    }
    
    //--------------------------------------------------------------------------
    /// view
    /// Create a sub view of the tensorArray relative to this view
    @inlinable
    func view(at offset: Shape.Tuple, extents: Shape.Tuple) -> Self {
        view(at: Shape.Array(offset), extents: Shape.Array(extents))
    }

    @inlinable
    func view(at offset: Shape.Array, extents: Shape.Array) -> Self {
        // the view created will have the same isShared state as the parent
        createView(at: offset, extents: extents,
                   strides: shape.strides, isReference: isShared)
    }

    //--------------------------------------------------------------------------
    /// view
    /// Create a sub view of the tensorArray relative to this view
    @inlinable
    func view(at offset: Shape.Tuple, extents: Shape.Tuple,
              strides: Shape.Tuple) -> Self
    {
        view(at: Shape.Array(offset), extents: Shape.Array(extents),
             strides: Shape.Array(strides))
    }
    
    @inlinable
    func view(at offset: Shape.Array, extents: Shape.Array,
              strides: Shape.Array) -> Self
    {
        // the view created will have the same isShared state as the parent
        createView(at: offset, extents: extents,
                   strides: strides, isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// viewItems
    /// Returns a view along the first dimension spanning all the others.
    /// It is used to simplify accessing a set of training samples.
    /// The view created will have the same isShared state as the parent
    @inlinable
    func viewItems(at offset: Int, count: Int) -> Self {
        // set starting offset
        var index = Shape.zeros
        index[0] = offset
        
        // set extent
        var viewExtents = Shape.zeros
        viewExtents[0] = count
        for i in 1..<viewExtents.count { viewExtents[i] = extents[i] }
        
        return createView(at: index, extents: viewExtents,
                          strides: shape.strides, isReference: isShared)
    }
    
    //--------------------------------------------------------------------------
    /// view(item:
    @inlinable
    func view(item: Int) -> Self {
        viewItems(at: item, count: 1)
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
            let size = Int(extents[0]) /
                ProcessInfo.processInfo.activeProcessorCount
            return size == 0 ? Int(extents[0]) : size
        }()
        let remainder = Int(extents[0]) % batchSize
        
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
        let lastBatchIndex = Int(extents[0]) - remainder
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
                count: shape.spanCount)
        }
    }
}

//==============================================================================
//
public extension TensorView where Element: FloatingPoint {
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

//==============================================================================
// Codable
public enum TensorCodingKeys: String, CodingKey { case extents, data }

public extension TensorView {
    /// encodes the contents of the array
    @inlinable
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: TensorCodingKeys.self)
        try container.encode(extents, forKey: .extents)
        try container.encode(tensorArray, forKey: .data)
    }
    
    @inlinable
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: TensorCodingKeys.self)
        let extents = try container.decode(Shape.Array.self, forKey: .extents)
        let array = try container.decode(TensorArray<Element>.self,
                                         forKey: .data)
        self = Self(shape: Shape(extents: extents), tensorArray: array,
                    viewOffset: 0, isShared: false)
    }
}

//==============================================================================
// == operator to simplify unit test syntax
public extension TensorView {
    /// compares the flat elements of self with a Swift array of elements
    @inlinable
    static func == (lhs: Self, rhs: [Element]) -> Bool {
        for (i, element) in lhs.elements().enumerated() {
            if element != rhs[i] { return false }
        }
        return true
    }
}

public extension TensorView where Element: AnyConvertable {
    /// compares the flat elements of self with a Swift collection of elements
    @inlinable
    static func == <R>(lhs: Self, rhs: R) -> Bool
        where R: Collection, R.Element: AnyConvertable
    {
        for (lhsElement, rhsElement) in zip(lhs.elements(), rhs) {
            if lhsElement != Element(any: rhsElement) { return false }
        }
        return true
    }

    /// compares the flat elements of self with a Swift collection of elements
    @inlinable
    static func == <T>(lhs: Self, rhs: T) -> Bool where T: AnyConvertable
    {
        lhs.element == Element(any: rhs)
    }

    /// compares the flat elements of self with a Swift collection of elements
    @inlinable
    static func == <T>(lhs: T, rhs: Self) -> Bool where T: AnyConvertable
    {
        Element(any: lhs) == rhs.element
    }
}
