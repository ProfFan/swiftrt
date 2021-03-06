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

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(test_Async.allTests),
        testCase(test_BinaryFunctions.allTests),
        testCase(test_Casting.allTests),
        testCase(test_Codable.allTests),
        testCase(test_Comparative.allTests),
        testCase(test_DataMigration.allTests),
        testCase(test_Initialize.allTests),
        testCase(test_IterateView.allTests),
        testCase(test_Math.allTests),
        testCase(test_Ranges.allTests),
        testCase(test_Reductions.allTests),
        testCase(test_Shape.allTests),
    ]
}
#endif
