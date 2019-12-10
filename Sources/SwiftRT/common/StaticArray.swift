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
    MutableCollection,
    CustomStringConvertible
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
    @inlinable @inline(__always)
    var count: Int {
        assert(MemoryLayout<Storage>.size % MemoryLayout<Element>.size == 0,
               "Storage size must be multiple of Element size")
        return MemoryLayout<Storage>.size / MemoryLayout<Element>.size
    }

    @inlinable @inline(__always)
    var startIndex: Int { 0 }

    @inlinable @inline(__always)
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
            withUnsafeBytes(of: storage) {
                $0.bindMemory(to: Element.self)[index]
            }
        }
        set {
            withUnsafeMutableBytes(of: &storage) {
                $0.bindMemory(to: Element.self)[index] = newValue
            }
        }
    }

    var description: String { String(describing: [Element](self)) }
}

//==============================================================================
// == operatpr
extension StaticArrayProtocol where Element: Equatable {
    @inlinable @inline(__always)
    public static func == (lhs: Self, rhs: Self) -> Bool {
        for i in 0..<lhs.count {
            if lhs[i] != rhs[i] { return false }
        }
        return true
    }
}

