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

package com.intel.analytics.bigdl.models

import java.util

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

@com.intel.analytics.bigdl.tags.Parallel
object Inception {
  def getModel[D: ClassTag](classNum: Int, modelName: String = "")(
    implicit ev: TensorNumeric[D]): Module[D] = {
    modelName match {
      case "inception-bn" =>
        def inception(inputSize: Int, config: Table)(
          implicit ev: TensorNumeric[D]): Module[D] = {
          val concat = Concat[D](2)
          if (config[Table](1)[Int](1) != 0) {
            val conv1 = Sequential[D]
            conv1.add(SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1))
            conv1.add(SpatialBatchNormalization(config[Table](1)(1), 1e-3))
            conv1.add(ReLU[D](true))
            concat.add(conv1)
          }

          val conv3 = Sequential[D]
          conv3.add(SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1))
          conv3.add(SpatialBatchNormalization(config[Table](2)(1), 1e-3))
          conv3.add(ReLU[D](true))
          conv3.add(SpatialConvolution[D](config[Table](2)(1),
            config[Table](2)(2), 3, 3, 1, 1, 1, 1))
          conv3.add(SpatialBatchNormalization(config[Table](2)(2), 1e-3))
          conv3.add(ReLU[D](true))
          concat.add(conv3)

          val conv3xx = Sequential[D]
          conv3xx.add(SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1))
          conv3xx.add(SpatialBatchNormalization(config[Table](3)(1), 1e-3))
          conv3xx.add(ReLU[D](true))

          conv3xx.add(SpatialConvolution[D](config[Table](3)(1),
            config[Table](3)(2), 3, 3, 1, 1, 1, 1))
          conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3))
          conv3xx.add(ReLU[D](true))

          conv3xx.add(SpatialConvolution[D](config[Table](3)(2),
            config[Table](3)(2), 3, 3, 1, 1, 1, 1))
          conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3))
          conv3xx.add(ReLU[D](true))
          concat.add(conv3xx)

          val pool = Sequential[D]
          pool.add(SpatialZeroPadding[D](1, 1, 1, 1))
          config[Table](4)[String](1) match {
            case "max" => pool.add(SpatialMaxPooling[D](3, 3, 1, 1).ceil())
            case "avg" => pool.add(SpatialAveragePooling[D](3, 3, 1, 1).ceil())
            case _ => throw new IllegalArgumentException
          }

          if (config[Table](4)[Int](2) != 0) {
            pool.add(SpatialConvolution[D](inputSize, config[Table](4)[Int](2), 1, 1, 1, 1))
            pool.add(SpatialBatchNormalization(config[Table](4)(2), 1e-3))
            pool.add(ReLU[D](true))
          }
          concat.add(pool)

          concat
        }
        val features = Sequential[D]
        features.add(SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3))
        features.add(SpatialBatchNormalization(64, 1e-3))
        features.add(ReLU[D](true))
        features.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil())
        features.add(SpatialConvolution[D](64, 64, 1, 1))
        features.add(ReLU[D](true))
        features.add(SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1))
        features.add(SpatialBatchNormalization(192, 1e-3))
        features.add(ReLU[D](true))
        features.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil())
        features.add(inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32))))
        features.add(inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64))))
        features.add(inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0))))
        features.add(SpatialConvolution[D](576, 576, 2, 2, 2, 2))
        features.add(inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128))))
        features.add(inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128))))
        features.add(inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96))))
        features.add(inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96))))

        val mainBranch = Sequential[D]
        mainBranch.add(inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0))))
        mainBranch.add(SpatialConvolution[D](1024, 1024, 2, 2, 2, 2))
        mainBranch.add(SpatialBatchNormalization(1024, 1e-3))
        mainBranch.add(inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128))))
        mainBranch.add(inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128))))
        mainBranch.add(SpatialAveragePooling[D](7, 7, 1, 1))
        mainBranch.add(View[D](1024).setNumInputDims(3))
        mainBranch.add(Linear[D](1024, classNum))
        mainBranch.add(LogSoftMax[D])

        val auxClassifier = Sequential[D]
        auxClassifier.add(SpatialAveragePooling[D](5, 5, 3, 3).ceil())
        auxClassifier.add(SpatialConvolution[D](576, 128, 1, 1, 1, 1))
        auxClassifier.add(SpatialBatchNormalization(128, 1e-3))
        auxClassifier.add(View[D](128 * 4 * 4).setNumInputDims(3))
        auxClassifier.add(Linear[D](128 * 4 * 4, 768))
        auxClassifier.add(ReLU[D](true))
        auxClassifier.add(Linear[D](768, classNum))
        auxClassifier.add(LogSoftMax[D])

        val splitter = Concat[D](2)
        splitter.add(mainBranch)
        splitter.add(auxClassifier)

        val model = Sequential[D]
        model.add(features)
        model.add(splitter)

        model
      case default =>
        val features = Sequential[D]
        features.add(SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3))
        features.add(ReLU[D](true))
        features.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil())
        features.add(SpatialConvolution[D](64, 64, 1, 1))
        features.add(ReLU[D](true))
        features.add(SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1))
        features.add(ReLU[D](true))
        features.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil())
        features.add(inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32))))
        features.add(inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64))))
        features.add(inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0))))
        features.add(SpatialConvolution[D](576, 576, 2, 2, 2, 2))
        features.add(inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128))))
        features.add(inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128))))
        features.add(inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96))))
        features.add(inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96))))

        val mainBranch = Sequential[D]
        mainBranch.add(inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0))))
        mainBranch.add(SpatialConvolution[D](1024, 1024, 2, 2, 2, 2))
        mainBranch.add(inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128))))
        mainBranch.add(inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128))))
        mainBranch.add(SpatialAveragePooling[D](7, 7, 1, 1))
        mainBranch.add(View[D](1024).setNumInputDims(3))
        mainBranch.add(Linear[D](1024, classNum))
        mainBranch.add(LogSoftMax[D])

        val auxClassifier = Sequential[D]
        auxClassifier.add(SpatialAveragePooling[D](5, 5, 3, 3).ceil())
        auxClassifier.add(SpatialConvolution[D](576, 128, 1, 1, 1, 1))
        auxClassifier.add(View[D](128 * 4 * 4).setNumInputDims(3))
        auxClassifier.add(Linear[D](128 * 4 * 4, 768))
        auxClassifier.add(ReLU[D](true))
        auxClassifier.add(Linear[D](768, classNum))
        auxClassifier.add(LogSoftMax[D])

        val splitter = Concat[D](2)
        splitter.add(mainBranch)
        splitter.add(auxClassifier)

        val model = Sequential[D]
        model.add(features)
        model.add(splitter)

        model
    }
  }

  def getGraphModel[D: ClassTag](classNum: Int, modelName: String = "")(
    implicit ev: TensorNumeric[D]): Module[D] = {
    modelName match {
      case "inception-bn" =>
        def inception(inputSize: Int, config: Table, inputNode: ModuleNode[D])(
          implicit ev: TensorNumeric[D]): ModuleNode[D] = {
          val concatInput = new ArrayBuffer[ModuleNode[D]]
          if (config[Table](1)[Int](1) != 0) {
            val conv1 = SpatialConvolution[D](inputSize,
              config[Table](1)(1), 1, 1, 1, 1).inputs(inputNode)
            val sb1 = SpatialBatchNormalization(config[Table](1)(1), 1e-3).inputs(conv1)
            val rl1 = ReLU[D](true).inputs(sb1)
            concatInput += rl1
          }

          val conv3 = SpatialConvolution[D](inputSize,
            config[Table](2)(1), 1, 1, 1, 1).inputs(inputNode)
          val sb3 = SpatialBatchNormalization(config[Table](2)(1), 1e-3).inputs(conv3)
          val rl3 = ReLU[D](true).inputs(sb3)
          val conv3_1 = SpatialConvolution[D](config[Table](2)(1),
            config[Table](2)(2), 3, 3, 1, 1, 1, 1).inputs(rl3)
          val sb3_1 = SpatialBatchNormalization(config[Table](2)(2), 1e-3).inputs(conv3_1)
          val rl3_1 = ReLU[D](true).inputs(sb3_1)
          concatInput += rl3_1


          val conv3xx = SpatialConvolution[D](inputSize,
            config[Table](3)(1), 1, 1, 1, 1).inputs(inputNode)
          val sb3xx = SpatialBatchNormalization(config[Table](3)(1), 1e-3).inputs(conv3xx)
          val rl3xx = ReLU[D](true).inputs(sb3xx)
          val conv3xx_1 = SpatialConvolution[D](config[Table](3)(1),
            config[Table](3)(2), 3, 3, 1, 1, 1, 1).inputs(rl3xx)
          val sb3xx_1 = SpatialBatchNormalization(config[Table](3)(2), 1e-3).inputs(conv3xx_1)
          val rl3xx_1 = ReLU[D](true).inputs(sb3xx_1)
          val conv3xx_2 = SpatialConvolution[D](config[Table](3)(2),
            config[Table](3)(2), 3, 3, 1, 1, 1, 1).inputs(rl3xx_1)
          val sb3xx_2 = SpatialBatchNormalization(config[Table](3)(2), 1e-3).inputs(conv3xx_2)
          val rl3xx_2 = ReLU[D](true).inputs(sb3xx_2)
          concatInput += rl3xx_2


          val padding = SpatialZeroPadding[D](1, 1, 1, 1).inputs(inputNode)
          var pool: ModuleNode[D] = null
          config[Table](4)[String](1) match {
            case "max" => pool = SpatialMaxPooling[D](3, 3, 1, 1).ceil().inputs(padding)
            case "avg" => pool = SpatialAveragePooling[D](3, 3, 1, 1).ceil().inputs(padding)
            case _ => throw new IllegalArgumentException
          }

          if (config[Table](4)[Int](2) != 0) {
            val conv4 = SpatialConvolution[D](inputSize,
              config[Table](4)[Int](2), 1, 1, 1, 1).inputs(pool)
            val sb4 = SpatialBatchNormalization(config[Table](4)(2), 1e-3).inputs(conv4)
            val rl4 = ReLU[D](true).inputs(sb4)
            concatInput += rl4
          } else {
            concatInput += pool
          }
          JoinTable[D](2, -1).inputs(concatInput: _*)
        }

        // feature
        val fconv1 = SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3).inputs()
        val fsb1 = SpatialBatchNormalization(64, 1e-3).inputs(fconv1)
        val frl1 = ReLU[D](true).inputs(fsb1)
        val smp = SpatialMaxPooling[D](3, 3, 2, 2).ceil().inputs(frl1)
        val fconv2 = SpatialConvolution[D](64, 64, 1, 1).inputs(smp)
        val frl2 = ReLU[D](true).inputs(fconv2)
        val fconv3 = SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1).inputs(frl2)
        val fsb3 = SpatialBatchNormalization(192, 1e-3).inputs(fconv3)
        val frl3 = ReLU[D](true).inputs(fsb3)
        val smp2 = SpatialMaxPooling[D](3, 3, 2, 2).ceil().inputs(frl3)
        val i1 = inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)), smp2)
        val i2 = inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)), i1)
        val i3 = inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)), i2)
        val conv5 = SpatialConvolution[D](576, 576, 2, 2, 2, 2).inputs(i3)
        val i4 = inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)), conv5)
        val i5 = inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)), i4)
        val i6 = inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)), i5)
        val i7 = inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)), i6)

        // Mainbranch
        val i8 = inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0)), i7)
        val mconv6 = SpatialConvolution[D](1024, 1024, 2, 2, 2, 2).inputs(i8)
        val msb = SpatialBatchNormalization(1024, 1e-3).inputs(mconv6)
        val i9 = inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)), msb)
        val i10 = inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)), i9)
        val msap = SpatialAveragePooling[D](7, 7, 1, 1).inputs(i10)
        val mview = View[D](1024).setNumInputDims(3).inputs(msap)
        val mlinear = Linear[D](1024, classNum).inputs(mview)
        val mls = LogSoftMax[D].inputs(mlinear)

        // AuxClassifier
        val asap = SpatialAveragePooling[D](5, 5, 3, 3).ceil().inputs(i7)
        val aconv = SpatialConvolution[D](576, 128, 1, 1, 1, 1).inputs(asap)
        val asb = SpatialBatchNormalization(128, 1e-3).inputs(aconv)
        val aview = View[D](128 * 4 * 4).setNumInputDims(3).inputs(asb)
        val alinear = Linear[D](128 * 4 * 4, 768).inputs(aview)
        val arelu = ReLU[D](true).inputs(alinear)
        val alinear2 = Linear[D](768, classNum).inputs(arelu)
        val als = LogSoftMax[D].inputs(alinear2)

        val endNode = JoinTable(2, -1).inputs(mls, als)

        val model = Graph(fconv1, endNode)

        model
      case default =>

        // features
        val fconv1 = SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3).inputs()
        val frl1 = ReLU[D](true).inputs(fconv1)
        val fsmp = SpatialMaxPooling[D](3, 3, 2, 2).ceil().inputs(frl1)
        val fconv2 = SpatialConvolution[D](64, 64, 1, 1).inputs(fsmp)
        val frl2 = ReLU[D](true).inputs(fconv2)
        val fconv3 = SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1).inputs(frl2)
        val frl3 = ReLU[D](true).inputs(fconv3)
        val fsmp2 = SpatialMaxPooling[D](3, 3, 2, 2).ceil().inputs(frl3)
        val i1 = inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32))).inputs(fsmp2)
        val i2 = inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64))).inputs(i1)
        val i3 = inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0))).inputs(i2)
        val fconv4 = SpatialConvolution[D](576, 576, 2, 2, 2, 2).inputs(i3)
        val i4 = inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128))).inputs(fconv4)
        val i5 = inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128))).inputs(i4)
        val i6 = inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96))).inputs(i5)
        val i7 = inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96))).inputs(i6)

        // mainBranch
        val mi8 = inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0))).inputs(i7)
        val mconv1 = SpatialConvolution[D](1024, 1024, 2, 2, 2, 2).inputs(mi8)
        val mi9 = inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128))).inputs(mconv1)
        val mi10 = inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128))).inputs(mi9)
        val msap = SpatialAveragePooling[D](7, 7, 1, 1).inputs(mi10)
        val mview = View[D](1024).setNumInputDims(3).inputs(msap)
        val mlinear = Linear[D](1024, classNum).inputs(mview)
        val mls = LogSoftMax[D].inputs(mlinear)

        // auxClassifier
        val asap = SpatialAveragePooling[D](5, 5, 3, 3).ceil().inputs(i7)
        val aconv1 = SpatialConvolution[D](576, 128, 1, 1, 1, 1).inputs(asap)
        val aview = View[D](128 * 4 * 4).setNumInputDims(3).inputs(aconv1)
        val alinear = Linear[D](128 * 4 * 4, 768).inputs(aview)
        val arl = ReLU[D](true).inputs(alinear)
        val alinear2 = Linear[D](768, classNum).inputs(arl)
        val als = LogSoftMax[D].inputs(alinear2)

        val endNode = JoinTable(2, -1).inputs(mls, als)
        val model = Graph(fconv1, endNode)

        model
    }
  }


  def inception[D: ClassTag](inputSize: Int, config: Table)(
    implicit ev: TensorNumeric[D]): Module[D] = {
    val concat = Concat[D](2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = Sequential[D]
      conv1.add(SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1))
      conv1.add(ReLU[D](true))
      concat.add(conv1)
    }

    val conv3 = Sequential[D]
    conv3.add(SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1))
    conv3.add(ReLU[D](true))
    conv3.add(SpatialConvolution[D](config[Table](2)(1),
      config[Table](2)(2), 3, 3, 1, 1, 1, 1))
    conv3.add(ReLU[D](true))
    concat.add(conv3)

    val conv3xx = Sequential[D]
    conv3xx.add(SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1))
    conv3xx.add(ReLU[D](true))
    conv3xx.add(SpatialConvolution[D](config[Table](3)(1),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1))
    conv3xx.add(ReLU[D](true))
    conv3xx.add(SpatialConvolution[D](config[Table](3)(2),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1))
    conv3xx.add(ReLU[D](true))
    concat.add(conv3xx)

    val pool = Sequential[D]
    pool.add(SpatialZeroPadding[D](1, 1, 1, 1))
    config[Table](4)[String](1) match {
      case "max" => pool.add(SpatialMaxPooling[D](3, 3, 1, 1).ceil())
      case "avg" => pool.add(SpatialAveragePooling[D](3, 3, 1, 1).ceil())
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(SpatialConvolution[D](inputSize, config[Table](4)[Int](2), 1, 1, 1, 1))
      pool.add(ReLU[D](true))
    }
    concat.add(pool)

    concat
  }

  def getModelCaffe[D: ClassTag](classNum: Int)
    (implicit ev: TensorNumeric[D]): Module[D] = {
    def inception[D: ClassTag](inputSize: Int, config: Table)(
      implicit ev: TensorNumeric[D]): Module[D] = {
      val concat = Concat[D](2)
      val conv1 = Sequential[D]
      conv1.add(SpatialConvolution[D](inputSize,
        config[Table](1)(1), 1, 1, 1, 1).setInitMethod(weightInitMethod = Xavier))
      conv1.add(ReLU[D](true))
      concat.add(conv1)

      val conv3 = Sequential[D]
      conv3.add(SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1).
        setInitMethod(Xavier))
      conv3.add(ReLU[D](true))
      conv3.add(SpatialConvolution[D](config[Table](2)(1),
        config[Table](2)(2), 3, 3, 1, 1, 1, 1).setInitMethod(weightInitMethod = Xavier))
      conv3.add(ReLU[D](true))
      concat.add(conv3)

      val conv5 = Sequential[D]
      conv5.add(SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1).
        setInitMethod(Xavier))
      conv5.add(ReLU[D](true))
      conv5.add(SpatialConvolution[D](config[Table](3)(1),
        config[Table](3)(2), 5, 5, 1, 1, 2, 2).setInitMethod(weightInitMethod = Xavier))
      conv5.add(ReLU[D](true))
      concat.add(conv5)

      val pool = Sequential[D]
      pool.add(SpatialMaxPooling[D](3, 3, 1, 1, 1, 1))
      pool.add(SpatialConvolution[D](inputSize, config[Table](4)(1), 1, 1, 1, 1).
        setInitMethod(Xavier))
      concat.add(pool)

      concat
    }

    val features = Sequential[D]
    features.add(SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3)
      .setInitMethod(weightInitMethod = Xavier))
    features.add(ReLU[D](true))
    features.add(SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    features.add(SpatialCrossMapLRN[D](5, 0.0001, 0.75))
    features.add(SpatialConvolution[D](64, 64, 1, 1, 1, 1, 0, 0)
      .setInitMethod(weightInitMethod = Xavier))
    features.add(ReLU[D](true))
    features.add(SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier))
    features.add(ReLU[D](true))
    features.add(SpatialCrossMapLRN[D](5, 0.0001, 0.75))
    features.add(SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    features.add(inception(192, T(T(64), T(96, 128), T(16, 32), T(32))))
    features.add(inception(256, T(T(128), T(128, 192), T(32, 96), T(64))))
    features.add(SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    features.add(inception(480, T(T(192), T(96, 208), T(16, 48), T(64))))

    features.add(inception(512, T(T(160), T(112, 224), T(24, 64), T(64))))
    features.add(inception(512, T(T(128), T(128, 256), T(24, 64), T(64))))
    features.add(inception(512, T(T(112), T(144, 288), T(32, 64), T(64))))

    features.add(inception(528, T(T(256), T(160, 320), T(32, 128), T(128))))
    features.add(SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    features.add(inception(832, T(T(256), T(160, 320), T(32, 128), T(128))))
    features.add(inception(832, T(T(384), T(192, 384), T(48, 128), T(128))))
    features.add(SpatialAveragePooling[D](7, 7, 1, 1))
    features.add(Dropout[D](0.4))
    features.add(View[D](1024).setNumInputDims(3))
    features.add(Linear[D](1024, classNum).setInitMethod(weightInitMethod = Xavier))
    features.add(LogSoftMax[D])
    features.reset()
    features
  }

  def performanceDouble(batchSize: Int, iter: Int, netType: String): Unit = {
    val input = Tensor[Double](batchSize, 3, 224, 224).fill(0.5)
    val model = getModelCaffe[Double](1000)
    val criterion = ClassNLLCriterion[Double]()
    var i = 0
    val sgd = new SGD[Double]
    val labelData = new Array[Double](batchSize)
    util.Arrays.fill(labelData, 10)
    val labels = Tensor[Double](Storage(labelData))

    println(model)
    println("warm up")
    while (i < 5) {
      val output = model.forward(input)
      val loss = criterion.forward(output, labels)
      val gradOutput = criterion.backward(output, labels)
      model.backward(input, gradOutput)
      i += 1
    }
    println("warm up done")
    model.resetTimes()
    var forwardTime = 0L
    var backwardTime = 0L
    while (i < iter) {
      var start = System.nanoTime()
      val output = model.forward(input)
      val loss = criterion.forward(output, labels)
      forwardTime += System.nanoTime() - start
      start = System.nanoTime()
      val gradOutput = criterion.backward(output, labels)
      model.backward(input, gradOutput)
      backwardTime += System.nanoTime() - start
      i += 1
    }
    println(s"forward time is ${forwardTime / iter / 1e6}ms")
    println(s"backward time is ${backwardTime / iter / 1e6}ms")
    val times = model.getTimes()
    var n = 0
    println(times.map(t => ( {
      n += 1;
      s"${t._1}-$n"
    }, (t._2 + t._3) / 1e9 / iter,
      t._2 / 1e9 / iter, t._3 / 1e9 / iter))
      .sortWith(_._2 > _._2).mkString("\n"))
  }

  def performanceFloat(batchSize: Int, iter: Int, netType: String): Unit = {
    val input = Tensor[Float](batchSize, 3, 224, 224).fill(0.5f)
    val model = getModelCaffe[Float](1000)
    val criterion = ClassNLLCriterion[Float]()
    var i = 0
    val sgd = new SGD[Float]
    val labelData = new Array[Float](batchSize)
    util.Arrays.fill(labelData, 10)
    val labels = Tensor[Float](Storage(labelData))

    println(model)
    println("warm up")
    while (i < 5) {
      val output = model.forward(input)
      val loss = criterion.forward(output, labels)
      val gradOutput = criterion.backward(output, labels)
      model.backward(input, gradOutput)
      i += 1
    }
    println("warm up done")
    model.resetTimes()
    var forwardTime = 0L
    var backwardTime = 0L
    while (i < iter) {
      var start = System.nanoTime()
      val output = model.forward(input)
      val loss = criterion.forward(output, labels)
      forwardTime += System.nanoTime() - start
      start = System.nanoTime()
      val gradOutput = criterion.backward(output, labels)
      model.backward(input, gradOutput)
      backwardTime += System.nanoTime() - start
      i += 1
    }
    val times = model.getTimes()
    var n = 0
    println(times.map(t => ( {
      n += 1;
      s"${t._1}-$n"
    }, (t._2 + t._3) / 1e9 / iter,
      t._2 / 1e9 / iter, t._3 / 1e9 / iter))
      .sortWith(_._2 > _._2).mkString("\n"))
    println(s"forward time is ${forwardTime / iter / 1e6}ms")
    println(s"backward time is ${backwardTime / iter / 1e6}ms")
    println(s"total time is ${(forwardTime + backwardTime) / iter / 1e6}ms")
  }

  def main(args: Array[String]): Unit = {
    require(args.length >= 1)
    args(0) match {
      case "perf" => args(3) match {
        case "double" => performanceDouble(args(1).toInt, args(2).toInt, "default")
        case "float" => performanceFloat(args(1).toInt, args(2).toInt, "default")
        case _ => throw new IllegalArgumentException
      }
      case _ => throw new IllegalArgumentException
    }
    System.exit(0)
  }
}
