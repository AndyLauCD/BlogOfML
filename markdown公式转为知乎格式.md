&ensp;&ensp;&ensp;&ensp;在知乎中写技术类文章，经常会用到markdown知乎文章可以导入markdown格式，但是不支持Latex公式。知乎大神提供了替代方案： https://zhuanlan.zhihu.com/p/69142198 

> 替换为：```\n<img src="https://www.zhihu.com/equation?tex=\1" alt="\1" class="ee_img tr_noresize" eeimg="1">\n```
>
> 查找目标：```\$\n*(.*?)\n*\$```
>
> 替换为：```\n<img src="https://www.zhihu.com/equation?tex=\1" alt="\1" class="ee_img tr_noresize" eeimg="1">\n```

&ensp;&ensp;&ensp;&ensp;为实现自动化，用python将其简易实现，代码如下：

```python
import re
import sys
def replace(file_name, output_file_name):
    try:
        pattern1 = r"\$\$\n*([\s\S]*?)\n*\$\$"
        new_pattern1 = r'\n<img src="https://www.zhihu.com/equation?tex=\1" alt="\1" class="ee_img tr_noresize" eeimg="1">\n'
        pattern2 = r"\$\n*(.*?)\n*\$"
        new_pattern2 =r'\n<img src="https://www.zhihu.com/equation?tex=\1" alt="\1" class="ee_img tr_noresize" eeimg="1">\n'
        f = open(file_name, 'r')
        f_output = open(output_file_name, 'w')
        all_lines = f.read()
        new_lines1 = re.sub(pattern1, new_pattern1, all_lines)
        new_lines2 = re.sub(pattern2, new_pattern2, new_lines1)
        f_output.write(new_lines2)
        # for line in all_lines:
        #     new_line1 = re.sub(pattern1, new_pattern1, line)
        #     new_line2 = re.sub(pattern2, new_pattern2, new_line1)
        #     f_output.write(new_line2)
        f.close()
        f_output.close()
    except Exception, e:
        print(e)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("need file name")
        sys.exit(1)
    file_name = sys.argv[1]
    # file_name = "极大似然小结.md".decode('utf-8')
    file_name_pre = file_name.split(".")[0]
    output_file_name = file_name_pre + "_zhihu.md"
    replace(file_name, output_file_name)

```

&ensp;&ensp;&ensp;&ensp;由此完成自动化配置。