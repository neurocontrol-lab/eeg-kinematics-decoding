import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
const runtime=process.env.CODEX_PRESENTATION_RUNTIME;
if (!runtime) throw new Error('Set CODEX_PRESENTATION_RUNTIME to the bundled dependencies directory.');
const {Presentation,PresentationFile}=await import(pathToFileURL(`${runtime}/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs`).href);

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const skill=process.env.PRESENTATIONS_SKILL_DIR;
if (!skill) throw new Error('Set PRESENTATIONS_SKILL_DIR to the installed presentation skill directory.');
process.env.RUNTIME_NODE_MODULES=`${runtime}/node/node_modules`;
const run=process.argv[2] || `${root}/outputs/original-p1-gpu`;
const result=JSON.parse(await fs.readFile(`${run}/results.json`,'utf8'));
if(result.status!=='complete') throw new Error('Wait for both original models to finish before documenting final results.');
const build=`${root}/tmp/presentation_builds/original-baseline`;
await fs.mkdir(build,{recursive:true});
await fs.mkdir(`${root}/presentations/exports`,{recursive:true});
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(`${skill}/container_tools/artifact_tool_utils.mjs`));
const family=resolvePresentationFont();
console.log('FONT',family);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const ink='#152A3B', muted='#4B6071', blue='#146B8C', orange='#C56732';
const source=`Source: original hand_motion.ipynb, SHA256 ${result.notebook_sha256}. Run: ${path.basename(run)}. Values are from this local run, not saved notebook outputs.`;
function text(s,t,x,y,w,h,size=28,color=ink,bold=false){
 const o=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 o.text=t;o.text.style={typeface:family,fontSize:size,color,bold,autoFit:'none'};return o;
}
function slide(title,note=source){
 const s=p.slides.add();s.background.fill='#FAFCFD';
 text(s,title,64,44,1152,title.includes('\n')?160:85,44,ink,true);
 s.speakerNotes.textFrame.setText(note);
 text(s,String(p.slides.items.length),1180,665,40,30,18,muted);
 return s;
}
function paragraphs(s,items){
 items.forEach((t,i)=>text(s,t,72,155+i*120,1136,104,29));
}
function table(s,values,widths,top=150,height=420,size=24){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:64,top,width:1152,height,columnWidths:widths,values});
 t.borders.assign({style:'solid',fill:'#DCE5EC',width:1});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?ink:(r%2?'#FFFFFF':'#EEF4F7');
  cell.text.style={typeface:family,fontSize:size,color:r===0?'#FFFFFF':ink,bold:r===0};
 }
 return t;
}
let s=slide('EEG to kinematics\nOriginal notebook baseline');
text(s,'Model walkthrough and local training results',72,270,1110,80,34,blue);
text(s,'Participant P1  |  29 September 2026',72,390,1110,50,27,muted);
text(s,'Research with Prof. Slawomir Nasuto',72,465,1110,50,27,muted);
text(s,'Original model, preprocessing and split settings preserved',72,570,1110,55,25);

s=slide('What one prediction means');
table(s,[['Stage','Original notebook behavior'],['Input','128 EEG samples × 32 electrodes'],['Window stride','64 samples: 50% overlap'],['Target','Three values at index i + 128'],['P1 split','19,228 training windows; 4,807 test windows'],['Scaling','Global min–max to [−1, 1], before splitting']],[290,862],150,410,26);
text(s,'At the original dataset’s 500 Hz: 256 ms of input, then a target one sample later. The cleaned files omit sampling-rate metadata.',72,590,1136,65,23,muted);
s.speakerNotes.textFrame.setText(source+' Dataset sampling: Luciw et al. (2014), doi:10.1038/sdata.2014.47. The notebook comment says 512 Hz, but that is not independently established for the repackaged data. The three target channels remain unidentified; no finger or wrist labels are inferred.');

s=slide('Model 1: temporal and spatial CNN');
table(s,[['Operation','Output per window','What it learns'],['Temporal Conv2D: 8 × (64, 1)','128 × 32 × 8','Patterns within each electrode'],['Depthwise: (1, 32), multiplier 2','128 × 1 × 16','Weighted electrode combinations'],['Separable Conv2D: 16 × (16, 1)','128 × 1 × 16','Temporal filtering and feature mixing'],['Average pool: (4, 1)','32 × 1 × 16','Time-axis reduction'],['Flatten → Dense(3)','512 → 3','Regression from window features']],[470,260,422],148,430,23);
text(s,'Batch normalization follows each convolution; dropout = 0.25 follows spatial and separable stages.',72,595,1136,55,23,muted);

s=slide('The CNN has no nonlinear activations');
paragraphs(s,[
 'The original convolution and dense layers use linear activations. There is no ReLU, ELU or other nonlinear activation.',
 'At inference, batch normalization uses fixed statistics and dropout is off. The complete model is therefore an affine mapping: ŷ = Ax + b.',
 'Its layered filters still impose structure on the mapping. Training includes dropout and batch-dependent normalization, so optimization is not identical to ordinary linear regression.',
 'This behavior is preserved for the original run; adding activations would be a later model change.'
]);

s=slide('Model 2: CNN with bidirectional LSTM');
table(s,[['Operation','Output per window','Role'],['Conv1D: 32 filters, kernel 3, ReLU','128 × 32','Local patterns across all channels'],['Batch normalization; dropout 0.3','128 × 32','Training normalization and regularization'],['Bidirectional LSTM: 64 units/direction','128','Two summaries of the input sequence'],['Dropout 0.3 → Dense(64), ReLU','64','Nonlinear feature combination'],['Dropout 0.2 → Dense(3)','3','Three continuous target predictions']],[535,210,407],148,430,23);
text(s,'The output layer is linear, suitable for regression rather than class probabilities.',72,595,1136,55,23,muted);

s=slide('What the LSTM adds');
paragraphs(s,[
 'At each time step, an LSTM updates a memory state. Learned gates control what it retains, writes and exposes as output.',
 'One LSTM reads the 128 samples forward and another reads them backward. Their final 64-value summaries concatenate into 128 values.',
 'Both directions operate inside the observed input window. They do not access EEG beyond the prediction target.',
 'The states reset between windows by default. This model does not carry memory continuously across an entire recording.'
]);
s.speakerNotes.textFrame.setText(source+' Keras Bidirectional documentation: https://keras.io/api/layers/recurrent_layers/bidirectional/ ; notebook uses return_sequences=False and default stateful=False.');

s=slide('Original training protocol');
const a=result.models.cnn,b=result.models.cnn_bilstm;
table(s,[['Setting','CNN','CNN–BiLSTM'],['Epochs','15','20'],['Batch size','64','64'],['Optimizer / objective','Adam / MSE','Adam / MSE'],['Total parameters',a.parameters.toLocaleString(),b.parameters.toLocaleString()],['Local training time',`${(a.training_seconds/60).toFixed(2)} min`,`${(b.training_seconds/60).toFixed(2)} min`],['Split','Random windows, 80:20','Same split']],[420,366,366],145,420,26);
text(s,'TensorFlow 2.20.0 on CPU. The test partition also serves as validation during training. No early stopping or model checkpoint selection.',72,585,1136,75,23,muted);
s.speakerNotes.textFrame.setText(source+' Python 3.9.13, NumPy 2.0.2, SciPy 1.13.1, scikit-learn 1.6.1. The notebook seeds train_test_split(42) only, not initialization or training shuffle. Modern TensorFlow does not support native Windows CUDA after TF 2.10: https://www.tensorflow.org/install/pip . Runtime is fit() wall-clock duration, excluding data loading and final inference.');

s=slide('Results on the original test partition');
const rows=[['Model / target','MSE','MAE','R²']];
for(const [name,m] of [['CNN',a],['CNN–BiLSTM',b]])for(let i=0;i<3;i++)rows.push([`${name} / target ${i+1}`,m.mse[i].toFixed(5),m.mae[i].toFixed(5),m.r2[i].toFixed(4)]);
table(s,rows,[435,239,239,239],150,420,26);
text(s,'MSE and MAE are in normalized target units. These scores describe this run’s random-window split, not independent-series or unseen-subject generalization.',72,590,1136,70,23,muted);

for(const [name,m] of [['CNN',a],['CNN–BiLSTM',b]]){
 s=slide(`${name}: learning curves`);
 const chart=s.charts.add('line',{
  position:{left:80,top:140,width:1120,height:430},
  categories:m.history.loss.map((_,i)=>String(i+1)),
  series:[{name:'Training MSE',values:m.history.loss.map(v=>Number(v.toFixed(8))),line:{fill:blue,width:3}},{name:'Validation MSE',values:m.history.val_loss.map(v=>Number(v.toFixed(8))),line:{fill:orange,width:3}}],
  hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:22}},lineOptions:{smooth:false},
  xAxis:{title:'Epoch',textStyle:{fontSize:18}},
  yAxis:{title:'Mean squared error',min:0,numberFormatCode:'0.00',textStyle:{fontSize:18},majorGridlines:{fill:'#DCE5EC',width:1}},
  chartFill:'#FAFCFD',plotAreaFill:'#FAFCFD',
 });applyPresentationChartFont(chart,{fontFamily:family});
 text(s,'Validation uses the same partition reported as test. Training includes dropout; validation uses inference behavior, so their losses need not match.',72,595,1136,65,23,muted);
}

s=slide('How to interpret the metrics');
paragraphs(s,[
 'MSE squares each prediction error, giving large errors more weight. This is the quantity Adam minimizes during training.',
 'MAE reports the mean absolute error in normalized target units; it is easier to interpret as a typical error magnitude.',
 'R² compares squared error with predicting the test target’s mean. R² = 1 is perfect; 0 matches that reference; negative values are worse.',
 'The three output labels and their original units remain unverified. These results cannot yet support claims about specific fingers or wrist coordinates.'
]);

s=slide('Recorded limitations and later experiments');
paragraphs(s,[
 'Adjacent windows overlap, but the split is random. Training and test partitions can therefore contain shared EEG samples.',
 'Scaling uses all P1 data before splitting, and windows can cross concatenated series boundaries. The test set also supplies validation feedback.',
 'The original CNN lacks nonlinear activations. Later P2 demonstration cells use incompatible orientation, window length and scaling assumptions.',
 'After reviewing this run: establish target provenance, correct splitting and scaling, then compare corrected models under a separate experiment ID.'
]);
s.speakerNotes.textFrame.setText(source+' No corrections have been applied in this run. The P2 demonstration cells were inspected but are not part of the executed P1 training experiment. This deck does not claim cross-subject validation.');

const candidate=`${build}/candidate.pptx`;
await(await PresentationFile.exportPptx(p)).save(candidate);
const final=`${root}/presentations/exports/original_baseline_v3.pptx`;
const checked=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,
 pythonExecutable:`${runtime}/python/python.exe`,
 integrityValidatorPath:`${skill}/container_tools/inspect_presentation_package_integrity.py`,
 layoutValidatorPath:`${skill}/container_tools/inspect_presentation_layout_geometry.py`,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[2,3,5,7,8].flatMap(i=>['--require-native-table-slide',String(i)])],
 requiredNativeTableOwnerSlides:[2,3,5,7,8],requiredNativeChartOwnerSlides:[9,10],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[family]},
 verifyArtifactToolImport:true,receiptPath:`${build}/validation.json`});
console.log(JSON.stringify(checked));
for(let i=0;i<p.slides.items.length;i++){
 const preview=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(`${build}/slide-${i+1}.png`,new Uint8Array(await preview.arrayBuffer()));
}
console.log('COMPLETE',final);
