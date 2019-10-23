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
// DeviceQueue
/// A device queue is an asynchronous sequential list of commands to be
/// executed on the associated device. It is a class protocol treated
/// as an abstract device interface
public protocol DeviceQueue:
    ObjectTracking,
    Logger,
    DeviceErrorHandling,
    QueueIntrinsics
{
    //--------------------------------------------------------------------------
    /// options to use when creating queue events
    var defaultQueueEventOptions: QueueEventOptions { get }
    /// the device the queue is associated with
    var device: ComputeDevice { get }
    /// if `true` the queue will execute functions synchronous with the app
    /// it is `false` by default and used for debugging
    var executeSynchronously: Bool { get set }
    /// a unique id used to identify the queue
    var id: Int { get }
    /// a name used to identify the queue
    var name: String { get }
    /// the maximum time to wait for an operation to complete
    /// a value of 0 (default) will wait forever
    var timeout: TimeInterval? { get set }
    
    //--------------------------------------------------------------------------
    // synchronization functions
    /// creates a QueueEvent
    func createEvent(options: QueueEventOptions) throws -> QueueEvent
    /// queues a queue event op. When executed the event is signaled
    @discardableResult
    func record(event: QueueEvent) throws -> QueueEvent
    /// records an op on the queue that will perform a queue blocking wait
    /// when it is processed
    func wait(for event: QueueEvent) throws
    /// blocks the calling thread until the queue queue has completed all work
    func waitUntilQueueIsComplete() throws

    //--------------------------------------------------------------------------
    // data transport functions
    /// clears the array to zero
    func zero(array: DeviceArray) throws
    /// asynchronously copies the contents of another device array
    func copyAsync(to array: DeviceArray, from otherArray: DeviceArray) throws
    /// asynchronously copies the contents of an app memory buffer
    func copyAsync(to array: DeviceArray,
                   from hostBuffer: UnsafeRawBufferPointer) throws
    /// copies the contents to an app memory buffer asynchronously
    func copyAsync(to hostBuffer: UnsafeMutableRawBufferPointer,
                   from array: DeviceArray) throws
    
    //--------------------------------------------------------------------------
    // debugging functions
    /// simulateWork(x:timePerElement:result:
    /// introduces a delay in the queue by sleeping a duration of
    /// x.shape.elementCount * timePerElement
    func simulateWork<T>(x: T, timePerElement: TimeInterval, result: inout T)
        where T: TensorView
    /// causes the queue to sleep for the specified interval for testing
    func delayQueue(atLeast interval: TimeInterval)
    /// for unit testing. It's part of the class protocol so that remote
    /// queues throw the error remotely.
    func throwTestError()
}

public extension DeviceQueue {
    func createEvent() throws -> QueueEvent {
        return try createEvent(options: defaultQueueEventOptions)
    }
}

let queueThreadViolationMessage =
    "a queue can only be accessed by the thread that created it"

//==============================================================================
/// LocalDeviceQueue
public protocol LocalDeviceQueue: DeviceQueue { }

public extension LocalDeviceQueue {
    //--------------------------------------------------------------------------
    /// handleDevice(error:
    func handleDevice(error: Error) {
        if (deviceErrorHandler?(error) ?? .propagate) == .propagate {
            device.handleDevice(error: error)
        }
    }
}

