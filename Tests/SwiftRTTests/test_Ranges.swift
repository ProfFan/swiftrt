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
        ("test_StridedRangeInForLoop", test_StridedRangeInForLoop),
        ("test_VectorRangeGradient", test_VectorRangeGradient),
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

        // sliding window starting at 2 and extending 3 (i.e 2 + 3)
        XCTAssert(vector[2..|3] == 2...4)

        // sliding window starting at 2 and extending 5, stepped
        XCTAssert(vector[2..|5..2] == [2, 4])
    }

    //==========================================================================
    // test_StridedRangeInForLoop
    func test_StridedRangeInForLoop() {
        XCTAssert([Int](0..<12..3) == [0, 3, 6, 9])
        XCTAssert((0..<8..2).count == 4)
        XCTAssert((0.0..<2.0..0.5).count == 4)
        XCTAssert([Double](0.0..<2.0..0.5) == [0.0, 0.5, 1.0, 1.5])
    }
    
    //==========================================================================
    // test_VectorRangeGradient
    func test_VectorRangeGradient() {
        let v = Vector(with: 0..<10)

        // simple range selection
        XCTAssert(gradientIsValid(at: v[1..<3], tolerance: 0.7, in: { exp($0) }))

        // test expression gradient
        let derivatives = v[2...] - v[1..<-1]
        XCTAssert(gradientIsValid(at: derivatives, tolerance: 0.7, in: { exp($0) }))
    }

    //==========================================================================
    // test_VectorSteppedRange
    func test_VectorSteppedRange() {
        let vector = IndexVector(with: 0...9)
        XCTAssert(vector[1..<2..2] == [1])
        XCTAssert(vector[1..<4..2] == [1, 3])
        XCTAssert(vector[..<4..2] == [0, 2])
        XCTAssert(vector[1...4..2] == [1, 3])
        XCTAssert(vector[1..<5..3] == [1, 4])
        XCTAssert(vector[1..<6..3] == [1, 4])
        XCTAssert(vector[..<8..3] == [0, 3, 6])
        XCTAssert(vector[1..<8..3] == [1, 4, 7])
        XCTAssert(vector[(...)..3] == [0, 3, 6, 9])
        XCTAssert(vector[(1...)..3] == [1, 4, 7])
    }

    //==========================================================================
    // test_MatrixRange
    func test_MatrixRange() {
        let m1 = IndexMatrix(3, 4, with: [
            0, 1,  2,  3,
            4, 5,  6,  7,
            8, 9, 10, 11
        ])
        
        let v1 = m1[1..<-1, ...3]
        XCTAssert(v1 == 4...6)

        // use negative row value to work from end and select row 1
        XCTAssert(m1[-2..<2, 1..<4] == 5...7)
        
        // sliding window starting at 0 and extending 2
        XCTAssert(m1[0..|2, ...] == [
            0, 1,  2,  3,
            4, 5,  6,  7,
        ])
    }

    //--------------------------------------------------------------------------
    // test_VectorWriteRange
    func test_VectorWriteRange() {
        var v1 = Vector(with: 0...6)
        let sevens = Vector(with: repeatElement(7, count: 3))
        v1[2...4] = sevens
        XCTAssert(v1 == [0, 1, 7, 7, 7, 5, 6])
        
        let m2 = Vector(with: 1...6)
        XCTAssert(gradientIsValid(at: m2, tolerance: 0.7, in: { exp($0) }))
    }
}
