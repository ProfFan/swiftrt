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
//
public struct StaticArray<Element, Storage> :
    RandomAccessCollection,
    MutableCollection
{
    //--------------------------------------------------------------------------
    // properties
    public var storage: Storage
    public let count: Int = {
        MemoryLayout<Storage>.size / MemoryLayout<Element>.size }()
    public let startIndex: Int = 0
    public var endIndex: Int { count }

    //--------------------------------------------------------------------------
    // initializers
    public init(data: Storage) {
        storage = data
    }

    public init?(data: Storage?) {
        guard let data = data else { return nil }
        self.init(data: data)
    }
    
    //--------------------------------------------------------------------------
    // indexing
    @inlinable @inline(__always)
    public subscript(index: Int) -> Element {
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
extension StaticArray: Equatable where Element: Equatable {
    public static func == (lhs: StaticArray<Element, Storage>, rhs: StaticArray<Element, Storage>) -> Bool {
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

//==============================================================================
//
public typealias StaticArray2<T> = StaticArray<T, (T, T)> where T: Equatable
