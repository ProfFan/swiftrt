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
//
public protocol StaticArrayProtocol :
    RandomAccessCollection,
    MutableCollection
{
    associatedtype Element
    associatedtype Storage
    
    var storage: Storage { get set }
    var count: Int { get }
    var startIndex: Int { get }
    var endIndex: Int { get }

    init(_ data: Storage)
    init?(_ data: Storage?)
}

//==============================================================================
//
public extension StaticArrayProtocol {
    // properties
    var count: Int {
        assert(MemoryLayout<Storage>.size % MemoryLayout<Element>.size == 0,
               "Storage size must be multiple of Element size")
        return MemoryLayout<Storage>.size / MemoryLayout<Element>.size
    }
    var startIndex: Int { 0 }
    var endIndex: Int { count }

    //--------------------------------------------------------------------------
    // initializers
    @inlinable @inline(__always)
    init?(_ data: Storage?) {
        guard let data = data else { return nil }
        self.init(data)
    }
    
    //--------------------------------------------------------------------------
    // indexing
    @inlinable @inline(__always)
    subscript(index: Int) -> Element {
        get {
            assert(index >= 0 && index < count, "index out of range")
            return withUnsafeBytes(of: storage) {
                $0.bindMemory(to: Element.self)[index]
            }
        }
        set {
            assert(index >= 0 && index < count, "index out of range")
            return withUnsafeMutableBytes(of: &storage) {
                $0.bindMemory(to: Element.self)[index] = newValue
            }
        }
    }
}

//==============================================================================
//
public struct StaticArray<Element, Storage> : StaticArrayProtocol {
    public var storage: Storage
    public init(_ data: Storage) {
        storage = data
    }
}

//==============================================================================
// == operatpr
extension StaticArrayProtocol where Element: Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        for i in 0..<lhs.count {
            if lhs[i] != rhs[i] { return false }
        }
        return true
    }
}


//==============================================================================
//
extension StaticArray: Codable where Element: Codable {
    enum CodingKeys: String, CodingKey { case name, data }

    /// encodes the contents of the array
    public func encode(to encoder: Encoder) throws {
//        var container = encoder.container(keyedBy: CodingKeys.self)
//        try container.encode(name, forKey: .name)
//        let buffer = try readOnly(using: DeviceContext.hostQueue)
//        try container.encode(ContiguousArray(buffer), forKey: .data)
    }
    
    public init(from decoder: Decoder) throws {
//        let container = try decoder.container(keyedBy: CodingKeys.self)
//        let name = try container.decode(String.self, forKey: .name)
//        let data = try container.decode(ContiguousArray<Element>.self,
//                                        forKey: .data)
//        self.init(elements: data, name: name)
        fatalError()
    }
}
