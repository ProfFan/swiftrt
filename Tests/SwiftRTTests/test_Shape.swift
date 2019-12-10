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

class test_Shape: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_perfDataShape2", test_perfDataShape2),
        ("test_perfShape2", test_perfShape2),
    ]

    //--------------------------------------------------------------------------
    // test_perfDataShape2
    func test_perfDataShape2() {
        #if !DEBUG
        var shape = DataShape()
        var anchor = true
        self.measure {
            for _ in 0..<100000 {
                let a = DataShape(extents: [3, 4])
                let b = a.columnMajor()
                let ds = a == b ? b.dense : a.dense
                let c = DataShape(extents: a.makePositive(indices: [1, -1]))
                let r = DataShape(extents: [1, 1]).repeated(to: a.extents)
                let j = a.joined(with: [ds, c, r], alongAxis: 1)
                let t = j.transposed()
                let li = t.linearIndex(of: [3, 2])
                let c0 = t.contains(shape: a)
                let c1 = t.contains(offset: [1, 1], extents: [2, 2])
                anchor = li > 0 && c0 && c1
                shape = t
            }
        }
        XCTAssert(anchor && shape.extents == [13, 3])
        #endif
    }

    //--------------------------------------------------------------------------
    // test_perfShape2
    func test_perfShape2() {
        #if !DEBUG
        var shape = Shape2(extents: Shape2.zeros)
        var anchor = true
        self.measure {
            for _ in 0..<100000 {
                let a = Shape2(extents: (3, 4))
                let b = a.columnMajor
                let ds = a == b ? b.dense : a.dense
                let c = Shape2(extents: Shape2.makePositive(dims: Shape2.Array((1, -1))))
                let r = Shape2(extents: Shape2.ones).repeated(to: a.extents)
                let j = a.joined(with: [ds, c, r], alongAxis: 1)
                let t = j.transposed()
                let li = t.linearIndex(of: (3, 2))
                let c0 = t.contains(other: a)
                let c1 = t.contains(offset: (1, 1), extents: (2, 2))
                anchor = li > 0 && c0 && c1
                shape = t
            }
        }
        XCTAssert(anchor && shape.extents == Shape2.Array((13, 3)))
        #endif
    }
}