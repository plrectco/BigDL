/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag

class ShareMemSpec extends FlatSpec with Matchers{
  def getFooModel[T: ClassTag]()(implicit ev: TensorNumeric[T]): Module[T] = {
    val feat1 = Sequential[T]()
    val feat2 = Sequential()

    feat1.add(MaskedSelect[T]())
    feat2.add(MaskedSelect[T]())


    feat1.add(Reshape[T](Array(4, 3, 6, 6)))
    feat2.add(Reshape[T](Array(4, 3, 6, 6)))


    feat1.add(SpatialDivisiveNormalization[T](3, Tensor.ones[T](3, 3)))
    feat1.add(SpatialFullConvolution[Tensor[T], T](3, 6, 3, 3))
    feat1.add(SpatialConvolution[T](6, 4, 5, 5))
    feat1.add(SpatialMaxPooling[T](2, 2, 2, 2))
    feat1.add(Max[T](2))
    feat1.add(Min[T](1))


    feat2.add(SpatialConvolution[T](3, 3, 5, 5))
    feat2.add(Min[T](2))
    feat2.add(Max[T](1))
    feat2.add(Linear[T](2, 2))
    feat2.add(CMul[T](Array(1, 2)))

    val model = Sequential[T]()
    val ct = ConcatTable[Tensor[T], T]()
    ct.add(feat1).add(feat2)

    model.add(ct)
    model.add(MixtureTable[T]())
    model.add(Add[T](2))
    model.add(Euclidean[T](2, 2))

    model
  }


  // TODO: Modules
  //    s.add(RoiPooling[T](1, 1))
  //    s.add(LookupTable[T](1, 2))





  "ShareMemSpec" should "works fine" in {
//    val  input = Tensor[Double](4, 3, 6, 6).fill(0.5)
    val input = T(Tensor[Double](4, 3, 6, 6).fill(0.5), Tensor[Double](4, 3, 6, 6).fill(1))
    val model = getFooModel[Double]()
    val model2 = model.cloneModule()

    val f1 = model.forward(input)
    val b1 = model.backward(input, f1.toTensor[Double].clone())

    model2.shareMem()
    val f2 = model2.forward(input)
    val b2 = model2.backward(input, f2.toTensor[Double].clone())

    f1 should be(f2)
    b1 should be(b2)
    println("Finished successfully")
  }

}