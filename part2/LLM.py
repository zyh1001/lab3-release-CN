import re
from Maze import Maze
from openai import OpenAI


# TODO: Replace this with your own prompt.
your_prompt = """
你是一个正在玩吃豆人游戏的最优策略AI。你的终极目标是以尽量少的步数吃掉地图上所有的豆子，并避免与鬼魂相撞。你的总步数应该在50步之内

下面是环境与规则说明：
1. 坐标系为(row, column)。目前地图中0代表空地，1代表墙壁，2代表豆子。
2. 动作对应的坐标变化：
   - 0 (上) -> 移动到 (row-1, column)
   - 1 (下) -> 移动到 (row+1, column)
   - 2 (左) -> 移动到 (row, column-1)
   - 3 (右) -> 移动到 (row, column+1)
3. 鬼魂的策略：本关卡中鬼魂不会移动，因此只需躲避：任何情况下，移动后的新坐标绝不能跟鬼魂位置重合，否则就会输掉游戏！
4. 在任何情况下，不能碰到墙壁，否则视为游戏失败。
4. 防止死循环：“曾经走过的位置”表明你的历史轨迹。除非进入死胡同后必须原路返回，否则请优先向着没走过的豆子区域移动。

你可以按照下面的步骤进行决策：
1. 全局扫描：阅读当前的迷宫布局，分析吃豆人、鬼魂所在位置，并标记所有未被吃掉的“豆子”位置。
2. 目标定位：通过评估路径（考虑避开墙壁和鬼魂），找出适合的豆子作为当前目标。
3. 动作评估：逐一分析提供的【可用方向】，推导每个方向对应走一步后的坐标。你可以考虑以下几个方面：
   （1）安全性验证：新坐标绝不能是鬼魂的领地！
   （2） 有效性验证：新坐标是否拉近了与目标豆子的距离？
   （3） 探索性验证：新坐标是否走过？（优先走没走过的新坐标向豆子靠近）
4. 综合以上因素，从可用方向中选出一个最佳的数字动作输出。
"""

# Don't change this part.
output_format = """
输出必须是0-3的整数，上=0，下=1，左=2，右=3。
*重点*：(5,5)的上方是(4,5)，下方是(6,5)，左方是(5,4)，右方是(5,6)。
输出格式为：
“分析：XXXX。
动作：0（一个数字，不能出现其他数字）。”
"""

prompt = your_prompt + output_format


def get_game_state(maze: Maze, places: list, available: list) -> str:
    """
    Convert game state to natural language description.
    """
    description = ""
    for i in range(maze.height):
        for j in range(maze.width):
            description += f"({i},{j})="
            if maze.grid[i, j] == 0:
                description += f"空地"
            elif maze.grid[i, j] == 1:
                description += "墙壁"
            else:
                description += "豆子"
            description += ","
        description += "\n"
    places_str = ','.join(map(str, places))
    available_str = ','.join(map(str, available))
    state = f"""当前游戏状态（坐标均以0开始）：\n1、迷宫布局（0=空地,1=墙,2=豆子）：\n{description}\n2、吃豆人位置：{maze.pacman_pos[4]}\n3、鬼魂位置：{maze.pacman_pos[3]}\n4、曾经走过的位置：{places_str}\n5、可用方向：{available_str}\n"""
    return state


def get_ai_move(client: OpenAI, model_name: str, maze: Maze, file, places: list, available: list) -> int:
    """
    Get the move from the AI model.
    :param client: OpenAI client instance.
    :param model_name: Name of the AI model.
    :param maze: The maze object.
    :param file: The log file to write the output.
    :param places: The list of previous positions.
    :param available: The list of available directions.
    :return: The direction chosen by the AI.
    """
    state = get_game_state(maze, places, available)

    file.write("________________________________________________________\n")
    file.write(f"message:\n{state}\n")
    print("________________________________________________________")
    print(f"message:\n{state}")

    print("Waiting for AI response...")
    all_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": state
            }
        ],
        stream=False,
        temperature=.0
    )
    info = all_response.choices[0].message.content

    file.write(f"AI response:\n{info}\n")
    print(f"AI response:\n{info}")
    numbers = re.findall(r'\d+', info)
    choice = numbers[-1]
    return int(choice), info
