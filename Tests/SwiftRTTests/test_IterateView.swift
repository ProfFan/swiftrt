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

class test_IterateView: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_Vector", test_Vector),
        ("test_Matrix", test_Matrix),
        ("test_MatrixRange", test_MatrixRange),
        ("test_MatrixSteppedRange", test_MatrixSteppedRange),
        ("test_Volume", test_Volume),
        ("test_VectorSubView", test_VectorSubView),
        ("test_MatrixSubView", test_MatrixSubView),
        ("test_VolumeSubView", test_VolumeSubView),
        ("test_perfVector", test_perfVector),
        ("test_perfMatrix", test_perfMatrix),
        ("test_perfVolume", test_perfVolume),
        ("test_perfIndexCopy", test_perfIndexCopy),
        ("test_repeatingValue", test_repeatingElement),
        ("test_repeatingRow", test_repeatingRow),
        ("test_repeatingCol", test_repeatingCol),
        ("test_repeatingColInVolume", test_repeatingColInVolume),
        ("test_repeatingMatrix", test_repeatingMatrix),
        ("test_repeatingMatrixSubView", test_repeatingMatrixSubView),
    ]
    
    //==========================================================================
    // test_Vector
    func test_Vector() {
        let count: Int32 = 10
        let expected = [Int32](0..<count)
        let vector = IndexVector(elements: expected)
        XCTAssert(vector.flatArray == expected)
    }
    
    //==========================================================================
    // test_Matrix
    func test_Matrix() {
        let expected = [Int32](0..<4)
        let matrix = IndexMatrix(2, 2, with: expected)
        XCTAssert(matrix.flatArray == expected)
    }
    
    //==========================================================================
    // test_MatrixRange
    func test_MatrixRange() {
        let m = IndexMatrix(2, 5, with: 0..<10)
        let v1 = m[(0, 1), (1, 3)].flatArray
        let expected1: [Int32] = [1, 2, 3, 6, 7, 8]
        XCTAssert(v1 == expected1)
        
        let m2 = IndexMatrix(5, 5, with: 0..<25)
        let v2 = m2[(1, 1), (3, 3)].flatArray
        let expected2: [Int32] = [6, 7, 8, 11, 12, 13, 16, 17, 18]
        XCTAssert(v2 == expected2)
    }
    
    //==========================================================================
    // test_MatrixSteppedRange
    func test_MatrixSteppedRange() {
        let m = IndexMatrix(2, 5, with: 0..<10)
        let v1 = m[(0, 1), (1, 3), (1, 2)].flatArray
        let expected1: [Int32] = [1, 3, 6, 8]
        XCTAssert(v1 == expected1)
    }
    
    //==========================================================================
    // test_Volume
    func test_Volume() {
        let expected = [Int32](0..<24)
        let volume = IndexVolume(2, 3, 4, with: expected)
        XCTAssert(volume.flatArray == expected)
    }

    //==========================================================================
    // test_VectorSubView
    func test_VectorSubView() {
        let vector = IndexVector(with: 0..<10)
        let view = vector.view(at: (2), extents: (3))
        let expected: [Int32] = [2, 3, 4]
        let values = view.flatArray
        XCTAssert(values == expected)
    }
    
    //==========================================================================
    // test_MatrixSubView
    func test_MatrixSubView() {
        let matrix = IndexMatrix(3, 4, with: 0..<12)
        let view = matrix.view(at: (1, 1), extents: (2, 2))
        let expected: [Int32] = [
            5, 6,
            9, 10
        ]
        XCTAssert(view.flatArray == expected)
    }
    
    //==========================================================================
    // test_VolumeSubView
    func test_VolumeSubView() {
        let volume = IndexVolume(3, 3, 4, with: 0..<36)
        let view = volume.view(at: (1, 1, 1), extents: (2, 2, 3))

        let expected: [Int32] = [
            17, 18, 19,
            21, 22, 23,

            29, 30, 31,
            33, 34, 35,
        ]
        XCTAssert(view.flatArray == expected)
    }
    
    //==========================================================================
    // test_perfVector
    func test_perfVector() {
        #if !DEBUG
        let count = 1024 * 1024
        let vector = IndexVector(with: 0..<count)
        //            print(matrix.formatted((2,0)))
        let values = vector.elements
        
        self.measure {
            for _ in values {}
        }
        #endif
    }
    
    //==========================================================================
    // test_perfMatrix
    func test_perfMatrix() {
        #if !DEBUG
        let rows = 1024
        let cols = 1024
        
        let matrix = Matrix<Int32>(rows, cols, with: 0..<(rows * cols))
        let values = matrix.elements
        
        self.measure {
            for _ in values {}
        }
        #endif
    }
    
    //==========================================================================
    // test_perfVolume
    func test_perfVolume() {
        #if !DEBUG
        let depths = 4
        let rows = 512
        let cols = 512
        
        let matrix = Volume<Int32>(depths, rows, cols,
                                   with: 0..<(depths * rows * cols))
        let values = matrix.elements
        
        self.measure {
            for _ in values {}
        }
        #endif
    }
    
    //==========================================================================
    // test_perfIndexCopy
    func test_perfIndexCopy() {
        #if !DEBUG
        var m = Matrix<Int32>(1024, 1024).startIndex
        
        self.measure {
            for _ in 0..<1000000 {
                m = m.increment()
//                m = m.advanced(by: 1)
            }
        }
        #endif
    }
    //==========================================================================
    // test_repeatingElement
    func test_repeatingElement() {
        let matrix = IndexMatrix(with: 42).repeated(to: (2, 3))
        let expected: [Int32] = [
            42, 42, 42,
            42, 42, 42,
        ]
        
        let values = matrix.flatArray
        XCTAssert(values == expected)
    }
    
    //==========================================================================
    // test_repeatingRow
    func test_repeatingRow() {
        let matrix = IndexMatrix(1, 3, with: 0...2).repeated(to: (2, 3))
        let expected: [Int32] = [
            0, 1, 2,
            0, 1, 2,
        ]
        
        let values = matrix.flatArray
        XCTAssert(values == expected)
    }
    
    //==========================================================================
    // test_repeatingCol
    func test_repeatingCol() {
        let matrix = IndexMatrix(3, 1, with: 0...2).repeated(to: (3, 2))
        let expected: [Int32] = [
            0, 0,
            1, 1,
            2, 2,
        ]
        XCTAssert(matrix.flatArray == expected)
    }
    
    //==========================================================================
    // test_repeatingColInVolume
    func test_repeatingColInVolume() {
        let v = IndexVolume(1, 3, 1, with: [1, 0, 1]).repeated(to: (2, 3, 4))
        let expected: [Int32] = [
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1,

            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1,
        ]
        XCTAssert(v.flatArray == expected)
    }
    
    //==========================================================================
    // test_repeatingMatrix
    func test_repeatingMatrix() {
        let volume = IndexVolume(with:
            [
                [
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 0, 1, 0],
                ],
            ]
        ).repeated(to: (2, 3, 4))

        let expected: [Int32] = [
            1, 0, 1, 0,
            0, 1, 0, 1,
            1, 0, 1, 0,

            1, 0, 1, 0,
            0, 1, 0, 1,
            1, 0, 1, 0,
        ]
        XCTAssert(volume.flatArray == expected)
    }
    
    //==========================================================================
    // test_repeatingMatrixSubView
    func test_repeatingMatrixSubView() {
        let matrix = IndexMatrix(3, 1, with: [1, 0, 1]).repeated(to: (3, 4))
        let expected: [Int32] = [
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1,
        ]
        XCTAssert(matrix.flatArray == expected)
        
        let view = matrix.view(at: (1, 1), extents: (2, 3))
        let viewExpected: [Int32] = [
            0, 0, 0,
            1, 1, 1,
        ]
        XCTAssert(view.flatArray == viewExpected)
    }
}
