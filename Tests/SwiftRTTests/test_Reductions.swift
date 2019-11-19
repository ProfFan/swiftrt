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

class test_Reductions: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_sumVector", test_sumVector),
        ("test_sumMatrix", test_sumMatrix),
        ("test_allVector", test_allVector),
        ("test_anyVector", test_anyVector),
        ("test_meanMatrix", test_meanMatrix),
    ]
    
    //--------------------------------------------------------------------------
    // test_sumVector
    func test_sumVector() {
        Platform.local.servicePriority = [cpuSynchronousServiceName]
        
        let m = Vector<Float>(with: [0, 1, 2, 3])
        let result = m.sum()
        XCTAssert(result.extents == [1])
        XCTAssert(result.element == 6)
    }

    //--------------------------------------------------------------------------
    // test_sumMatrix
    func test_sumMatrix() {
        Platform.local.servicePriority = [cpuSynchronousServiceName]

        let m = Matrix<Float>(3, 2, with: [
            0, 1,
            2, 3,
            4, 5
        ])

        // sum all
        do {
            let result = m.sum()
            XCTAssert(result.extents == [1, 1])
            XCTAssert(result.element == 15)
        }

        do {
            let result = m.sum(alongAxes: 0, 1)
            XCTAssert(result.extents == [1, 1])
            XCTAssert(result.element == 15)
        }
        
        // sum cols
        do {
            let result = m.sum(alongAxes: 1)
            let expected: [Float] = [
                1,
                5,
                9
            ]
            XCTAssert(result.extents == [3, 1])
            XCTAssert(result.flatArray == expected)
        }

        // sum rows
        do {
            let result = m.sum(alongAxes: 0)
            let expected: [Float] = [
                6, 9
            ]
            XCTAssert(result.extents == [1, 2])
            XCTAssert(result.flatArray == expected)
        }
    }

    //--------------------------------------------------------------------------
    // test_allVector
    func test_allVector() {
        Platform.local.servicePriority = [cpuSynchronousServiceName]
        
        do {
            let m = Vector(elements: [true, true, true])
            XCTAssert(m.all().element == true)
        }
        
        do {
            let m = Vector(elements: [true, false, true])
            XCTAssert(m.all().element == false)
        }

        do {
            let m = Vector(elements: [false, false, false])
            XCTAssert(m.all().element == false)
        }
    }
    
    //--------------------------------------------------------------------------
    // test_anyVector
    func test_anyVector() {
        Platform.local.servicePriority = [cpuSynchronousServiceName]
        
        do {
            let m = Vector(elements: [true, true, true])
            XCTAssert(m.any().element == true)
        }
        
        do {
            let m = Vector(elements: [false, false, true])
            XCTAssert(m.any().element == true)
        }
        
        do {
            let m = Vector(elements: [false, false, false])
            XCTAssert(m.any().element == false)
        }
    }
    
    //--------------------------------------------------------------------------
    // test_meanMatrix
    func test_meanMatrix() {
//        Platform.local.servicePriority = [cpuSynchronousServiceName]
//
//        let m = Matrix<Float>(3, 2, with: [
//            0, 1,
//            2, 3,
//            4, 5
//        ])
//
//        // mean all
//        do {
//            let result = m.mean()
//            XCTAssert(result.extents == [1, 1])
//            XCTAssert(result.element == 15 / 6)
//        }
//
//        do {
//            let result = m.mean(alongAxes: 0, 1)
//            XCTAssert(result.extents == [1, 1])
//            XCTAssert(result.element == 15 / 6)
//        }
//
//        // mean cols
//        do {
//            let result = m.mean(alongAxes: 1)
//            let expected: [Float] = [
//                0.5,
//                2.5,
//                4.5
//            ]
//            XCTAssert(result.extents == [3, 1])
//            XCTAssert(result.flatArray == expected)
//        }
//
//        // mean rows
//        do {
//            let result = m.mean(alongAxes: 0)
//            let expected: [Float] = [
//                3, 4.5
//            ]
//            XCTAssert(result.extents == [1, 2])
//            XCTAssert(result.flatArray == expected)
//        }
    }
    

}
