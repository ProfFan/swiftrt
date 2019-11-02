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
/// CpuComputeService
public protocol CpuComputeService: ComputeService {
    var configuration: [CpuPropertyKey: Any] { get set }
}

//==============================================================================
/// CpuService
public class CpuService: CpuComputeService, LocalComputeService {
    // properties
    public private(set) weak var platform: ComputePlatform!
    public private(set) var trackingId = 0
    public private(set) var devices = [ComputeDevice]()
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()
    public let id: Int
    public var logInfo: LogInfo
    public let name: String
        
    // configuration and defaults
    public var configuration: [CpuPropertyKey: Any] = [
        .queuesPerDevice: 2
    ]

    // timeout
    public var timeout: TimeInterval? {
        didSet {
            devices.forEach { $0.timeout = timeout }
        }
    }

    //--------------------------------------------------------------------------
    // initializers
    public required init(platform: ComputePlatform,
                         id: Int,
                         logInfo: LogInfo,
                         name: String? = nil) throws {
        self.platform = platform
        self.id = id
        self.name = name ?? cpuServiceName
        self.logInfo = logInfo
        
        // this is held statically by the Platform
        trackingId = ObjectTracker.global.register(self, isStatic: true)

        // add normal uma cpu device
        devices.append(CpuDevice(service: self,
                                 deviceId: 0,
                                 logInfo: logInfo,
                                 addressing: .unified,
                                 timeout: timeout))
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
}

//==============================================================================
/// a set of predefined property names to simplify configuring
/// the service properties
public enum CpuPropertyKey: Int {
    case queuesPerDevice
}

//==============================================================================
/// TestCpuService
/// A cpu implementation that acts like a discreet memory device
public class TestCpuService: CpuComputeService, LocalComputeService {
    // properties
    public private(set) weak var platform: ComputePlatform!
    public private(set) var trackingId = 0
    public private(set) var devices = [ComputeDevice]()
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()
    public let id: Int
    public var logInfo: LogInfo
    public let name: String
    
    // configuration and defaults
    public var configuration: [CpuPropertyKey: Any] = [
        .queuesPerDevice: 2
    ]
    
    // timeout
    public var timeout: TimeInterval? {
        didSet {
            devices.forEach { $0.timeout = timeout }
        }
    }
    
    //--------------------------------------------------------------------------
    // initializers
    public required init(platform: ComputePlatform,
                         id: Int,
                         logInfo: LogInfo,
                         name: String? = nil) throws {
        self.platform = platform
        self.id = id
        self.name = name ?? testCpuServiceName
        self.logInfo = logInfo
        
        // this is held statically by the Platform
        trackingId = ObjectTracker.global.register(self, isStatic: true)
        
        // add discreet cpu devices only for testing
        devices.append(CpuDevice(service: self,
                                 deviceId: 1,
                                 logInfo: logInfo,
                                 addressing: .discreet,
                                 timeout: timeout))
        
        devices.append(CpuDevice(service: self,
                                 deviceId: 2,
                                 logInfo: logInfo,
                                 addressing: .discreet,
                                 timeout: timeout))
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
}
