#include <catboost/libs/algo/error_functions.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/metrics/metric.h>

#include <library/unittest/registar.h>


Y_UNIT_TEST_SUITE(SoftFilterNumericDifferentiationTest) {
        Y_UNIT_TEST(OneFeatureTest) {
            double q=0.5;
            const auto LqMetric = MakeLqLogLossMetric(q);
            const auto lqLogLoss = new TLqLogLoss(q, false);

            const int numSamples = 50;
            const int numQueries = 1;
            const int samplesPerQuery = numSamples / numQueries;
            const double epsilon = 1e-6;

            TVector<TVector<double>> approx = {TVector<double>(numSamples, 0.f)};

            TFastRng32 rng = TFastRng32{42, 2};

            TVector<float> target(numSamples);
            Generate(target.begin(), target.end(), [&](){ return static_cast<float>(rng.GenRandReal1() > 0.5); });

            TVector<float> weight(numSamples);
            Generate(weight.begin(), weight.end(), [&](){ return rng.GenRandReal1();});

            TVector<TQueryInfo> queriesInfo(numQueries);
            for (int i = 0; i < numQueries; i++) {
                queriesInfo[i] = TQueryInfo{static_cast<ui32>(i * samplesPerQuery),
                                            static_cast<ui32>((i + 1) * samplesPerQuery)};
            }

            TVector<TDers> ders(numSamples);
            TVector<double> deltas(numSamples);

            NPar::TLocalExecutor executor;

            for (int i = 0; i < 100; i++) {
                Generate(approx[0].begin(), approx[0].end(), [&](){ return rng.GenRandReal1() * 4. - 2.; });

                auto metricHolder = LqMetric->Eval(approx, target, weight, queriesInfo, 0, queriesInfo.ysize(), executor);
                const double metric = metricHolder.Stats[0];
                Y_ASSERT(metricHolder.Stats[1] == 2.);

                lqLogLoss->CalcDersRange(0, 1, false, approx[0].data(), deltas.data(),
                        target.data(), weight.data(), ders.data());


                for (int j = 0; j < approx[0].ysize(); j++) {
                    approx[0][j] += epsilon;
                    auto shiftMetricHolder = LqMetric->Eval(approx, target, weight, queriesInfo, 0, queriesInfo.ysize(), executor);
                    const double shiftMetric = shiftMetricHolder.Stats[0];
                    Y_ASSERT(shiftMetricHolder.Stats[1] == 2.);

                    Cout << "origin, shift = [" << metric << ',' << shiftMetric << ']' << Endl;
                    const double numericDer = (shiftMetric - metric) / epsilon;
                    Cout << "val approx " << approx[0][j] << Endl;
                    Cout << "target " << target[j] << Endl;
                    Cout << "der " << ders[j].Der1 << Endl;
                    Cout << "numeric, analytic = [" << numericDer << ',' << ders[j].Der1 << "]" << Endl;
                    UNIT_ASSERT_DOUBLES_EQUAL(ders[j].Der1, numericDer, epsilon);

                    approx[0][j] -= epsilon;
                }
            }
        }
}