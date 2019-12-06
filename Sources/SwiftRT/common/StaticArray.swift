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

public protocol ZeroedStorage {
    init()
}

extension Int8: ZeroedStorage {}
extension Int16: ZeroedStorage {}
extension Int32: ZeroedStorage {}
extension Int: ZeroedStorage {}

//==============================================================================
//
public struct StaticArray<Storage> :
    RandomAccessCollection, MutableCollection, Equatable, Codable
    where Storage: ArrayStorage
{
    //--------------------------------------------------------------------------
    // properties
    public var storage: Storage
    public let startIndex: Int = 0
    public let count: Int = Storage.count
    public let endIndex: Int = Storage.count

    //--------------------------------------------------------------------------
    // initializers
    public init(_ data: Storage) {
        storage = data
    }

    public init?(_ data: Storage?) {
        guard let data = data else { return nil }
        self.init(data)
    }
    
    //--------------------------------------------------------------------------
    // indexing
    @inlinable @inline(__always)
    public subscript(index: Int) -> Storage.Element {
        get {
            assert(index >= 0 && index < count, "index out of range")
            return withUnsafeBytes(of: storage) {
                $0.bindMemory(to: Storage.Element.self)[index]
            }
        }
        set {
            assert(index >= 0 && index < count, "index out of range")
            return withUnsafeMutableBytes(of: &storage) {
                $0.bindMemory(to: Storage.Element.self)[index] = newValue
            }
        }
    }
}

//==============================================================================
//
public protocol ArrayStorage: ZeroedStorage, Equatable, Codable {
    associatedtype Element: ZeroedStorage, Equatable, Codable
    static var count: Int { get }
}

public extension ArrayStorage {
    static var count: Int { MemoryLayout<Self>.size / MemoryLayout<Element>.size }
}

public struct ArrayStorage2<Element>: ArrayStorage where
    Element: ZeroedStorage & Equatable & Codable
{
    private var _e0, _e1: Element
    typealias E = Element

    public init() { _e0 = E(); _e1 = E() }
    init(_ e0: E, _ e1: E) { _e0 = e0; _e1 = e1 }
}

//==============================================================================
//
public typealias StaticArray2<T> =
    StaticArray<ArrayStorage2<T>> where T: ZeroedStorage & Equatable & Codable
