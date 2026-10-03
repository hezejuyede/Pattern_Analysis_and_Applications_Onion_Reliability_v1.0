"""Render both main numerical figures from independently reconstructed split tables."""
import argparse,importlib.util,json
from pathlib import Path
import pandas as pd

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();output=args.out.resolve();output.mkdir(parents=True,exist_ok=True)
    # Reuse only the preserved presentation routine, with explicit new input/output.
    code=Path(__file__).resolve().parents[1]/'original/plot_joint_and_single.py'
    spec=importlib.util.spec_from_file_location('preserved_figure_rendering',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.OUT=output
    names=[('Fig_2_Joint_Replacement','joint'),('Fig_3_Single_Source_Replacement','single')]
    for name,design in names:
        data=pd.read_csv(args.tables/(name+'_split_source.csv'))
        module.render(data,design,name)
    (output/'RENDERING_SCOPE.json').write_text(json.dumps({'status':'RENDERED_FROM_RECONSTRUCTED_SPLIT_TABLES','source':str(args.tables.resolve()),
        'visual_review':'This command renders files; it does not itself certify visual layout.','interpretation':'Joint bars are 30-split 2.5th-97.5th percentiles; single bars are ten-split min/max. Neither is a biological confidence interval. Common-class weighting is a post hoc sensitivity defined after v17; it restricts evaluation classes for joint and target plus evaluation classes for single.'},indent=2)+'\n',encoding='utf-8')
    print('Rendered both main result figures from reconstructed split-level inputs.')
if __name__=='__main__':main()
