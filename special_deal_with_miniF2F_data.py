import glob


for file in glob.glob("/hpc2hdd/home/zyang398/miniF2F/isabelle/**/*.thy", recursive=True):
    print(file)
    with open(file, "r") as f:
        file_content = f.read()
    begin_idx = file_content.index("imports")
    end_idx = file_content.index("begin")

    file_content = file_content[:begin_idx] + '\nimports HOL.HOL Complex_Main "HOL-Library.Code_Target_Numeral" "HOL-Library.Sum_of_Squares" "HOL-Computational_Algebra.Computational_Algebra" "HOL-Number_Theory.Number_Theory"\n' + file_content[end_idx:] 
    with open(file, "w") as f:
        f.write(file_content)
    