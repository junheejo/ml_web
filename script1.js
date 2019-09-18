import {MnistData} from './data.js';

async function showExamples(data) {
  //바이저 내부에 컨테이너 생성
  const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});
  //examples 가져오기
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  //각 example을 실행 할 

}
