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
import XCTest
import Foundation

@testable import SwiftRT

class test_Ranges: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_VectorRange", test_VectorRange),
        ("test_VectorSteppedRange", test_VectorSteppedRange),
        ("test_VectorWriteRange", test_VectorWriteRange),
        ("test_MatrixRange", test_MatrixRange),
    ]
    
    //==========================================================================
    // test_VectorRange
    func test_VectorRange() {
        let vector = IndexVector(with: 0..<10)
        
        // from index 1 through the end
        XCTAssert(vector[1...] == 1...9)

        // through last element
        XCTAssert(vector[...-1] == 0...9)
        XCTAssert(vector[...] == 0...9)

        // up to the second to last element
        XCTAssert(vector[..<-2] == 0...7)

        // between 4 and 2 back from the end
        XCTAssert(vector[-4..<-2] == 6...7)

        // the whole range stepping by 2
        XCTAssert(vector[(...)..2] == 0..<10..2)
        XCTAssert(vector[.....2] == 0..<10..2)

        // the whole range stepping backwards by 2
//        let reversed = [Int32](stride(from: 0, to: 10, by: 2).reversed())
//        XCTAssert(vector[(...)..-2] == reversed)
//        XCTAssert(vector[.....-2] == reversed)

        // sliding window starting at 2 and extending 3 (i.e 2 + 3)
        XCTAssert(vector[2..|3] == 2...4)

        // sliding window starting at 2 and extending 5, stepped
        XCTAssert(vector[2..|5..2] == [2, 4])
    }

    //==========================================================================
    // test_VectorSteppedRange
    func test_VectorSteppedRange() {
        let vector = IndexVector(with: 0...9)
        let v1 = vector[(1), (2), (2)].flatArray
        XCTAssert(v1.count == 1)
        let v2 = vector[(1), (4), (2)].flatArray
        XCTAssert(v2.count == 2)
        let v3 = vector[(1), (4), (2)].flatArray
        XCTAssert(v3.count == 2)

        let v4 = vector[(1), (5), (3)].flatArray
        XCTAssert(v4.count == 2)
        let e4: [Int32] = [1, 4]
        XCTAssert(v4 == e4)

        let v5 = vector[(1), (6), (3)].flatArray
        XCTAssert(v5.count == 2)
        let e5: [Int32] = [1, 4]
        XCTAssert(v5 == e5)

        let v6 = vector[(1), (8), (3)].flatArray
        XCTAssert(v6.count == 3)
        let e6: [Int32] = [1, 4, 7]
        XCTAssert(v6 == e6)
    }

    //==========================================================================
    // test_MatrixRange
    func test_MatrixRange() {
        let m1 = IndexMatrix(3, 4, with: 0..<12)
        let v1 = m1[1..<-1, ...3]
        XCTAssert(v1 == 4...6)

//        // negative values work back from the end
//        let v2 = m1[(-1, 1), (2, 4)]
//        XCTAssert(v2.flatArray == [Int32](5...7))

        // range syntax
//        XCTAssert(vector[...].flatArray == [Int32](0...9))
//        XCTAssert(vector[1...].flatArray == [Int32](1...9))
    }

    //--------------------------------------------------------------------------
    // test_VectorWriteRange
    func test_VectorWriteRange() {
//        Platform.local.servicePriority = [cpuSynchronousServiceName]
//        Platform.log.level = .diagnostic
//        Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
//        var v1 = Vector(with: 0...6)
//        let sevens = Vector(with: repeatElement(7, count: 3))
//        v1[0..<2] = sevens
//        XCTAssert(v1.flatArray == [0, 1, 7, 7, 7, 5])
//        let m2 = Vector(with: 1...6)
//        XCTAssert(gradientIsValid(at: m2, tolerance: 0.7, in: { exp($0) }))
    }
}
