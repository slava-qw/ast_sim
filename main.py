import ast
from ast import parse, NodeTransformer, FunctionDef, ClassDef, copy_location
from collections import defaultdict
import numpy as np


class PrepVariables(NodeTransformer):
    def __init__(self):
        self.v_ids = defaultdict(str)
        super().__init__()

    def visit_Name(self, node: ast.Name):
        self.generic_visit(node)

        match self.v_ids[node.id]:
            case '':
                var_name = f'id_{len(self.v_ids)}'
            case _:
                var_name = self.v_ids[node.id]

        self.v_ids[node.id] = var_name if self.v_ids[node.id] != 'self' else 'self'

        new_name = ast.Name(id=var_name)
        return copy_location(new_name, node)

    def visit_BinOp(self, node: ast.BinOp):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)

        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            bin_ops = {
                ast.Add: lambda x, y: x + y,
                ast.Mult: lambda x, y: x * y,
                ast.Sub: lambda x, y: x - y,
                ast.Div: lambda x, y: x / y,
                ast.FloorDiv: lambda x, y: x // y,
                ast.Mod: lambda x, y: x % y,
                ast.Pow: lambda x, y: x ** y
            }

            try:
                new_val = bin_ops[type(node.op)](node.left.value, node.right.value)
                result = ast.Constant(new_val)
                return copy_location(result, node)
            except KeyError as e:
                return node

        return node


class PrepFunctions(NodeTransformer):
    def __init__(self):
        self.f_ids = defaultdict(str)
        self.v_ids = defaultdict(str)
        super().__init__()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)

        # delete unusable functions (which contain 'pass' in the body)
        if any([isinstance(object, ast.Pass) for object in node.body]):
            del node
            return

        # change the name of the function
        match self.f_ids[node.name]:
            case '':
                func_name = f'func_{str(len(self.f_ids))}'
            case _:
                func_name = self.f_ids[node.name]
        self.f_ids[node.name] = func_name

        # change the name of attribute
        for i, arg in enumerate(node.args.args):
            arg.arg = f'arg_{i}' if arg.arg != 'self' else 'self'

        # delete docstring from body of the function
        try:
            if isinstance(node.body[0].value, ast.Str) and isinstance(node.body[0], ast.Expr):
                del node.body[0]
        except AttributeError as e:
            pass  # thus, there is no docstring

        # change the node that corresponds to the function
        func_dict = {
            'name': func_name,
            'args': node.args,
            'body': node.body,
            'decorator_list': node.decorator_list
        }
        new_func = FunctionDef(**func_dict)

        # also I should delete all Expr objects from body

        return copy_location(new_func, node)


class PrepClasses(NodeTransformer):
    def __init__(self):
        self.c_ids = defaultdict(str)
        self.attr_ids = defaultdict(str)
        super().__init__()

    def visit_ClassDef(self, node: ast.ClassDef):
        self.generic_visit(node)

        # change the name of the class
        match self.c_ids[node.name]:
            case '':
                cls_name = f'cls_{str(len(self.c_ids))}'
            case _:
                cls_name = self.c_ids[node.name]
        self.c_ids[node.name] = cls_name

        class_dict = {
            'name': cls_name,
            'bases': node.bases,
            'keywords': node.keywords,
            'body': node.body,
            'decorator_list': node.decorator_list
        }
        new_cls = ClassDef(**class_dict)

        return copy_location(new_cls, node)

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)

        attr_name = f'attr_{len(self.attr_ids)}'
        self.attr_ids[node.attr] = attr_name
        new_attr = ast.Attribute(value=node.value, attr=attr_name)

        return copy_location(new_attr, node)


def body_sort(ast_tree):
    """Tried to implement an object order check"""
    for node in ast.walk(ast_tree):
        if hasattr(node, 'body'):
            # problems with order (see 4, 4.2)
            d = {'Assign': [[], lambda x: x],
                 'FunctionDef': [[], lambda x: x.name],
                 'ClassDef': [[], lambda x: x.name],
                 'another': [[], lambda x: x]
                 }

            for body_object in node.body:
                if isinstance(body_object, ast.Assign):
                    d['Assign'][0].append(body_object)
                elif isinstance(body_object, ast.FunctionDef):
                    d['FunctionDef'][0].append(body_object)
                elif isinstance(body_object, ast.ClassDef):
                    d['ClassDef'][0].append(body_object)
                else:
                    d['another'][0].append(body_object)

            list_of_lists = [sorted(d[name][0], key=d[name][1]) for name in d.keys()]
            node.body = [item for sublist in list_of_lists for item in sublist]


def prep_text_code(file_name: str) -> str:
    with open(file_name) as f:
        ast_tree = parse(f.read())

        ast_tree = PrepFunctions().visit(ast_tree)
        ast_tree = PrepClasses().visit(ast_tree)
        ast_tree = PrepVariables().visit(ast_tree)

        # sort ast-tree (it raises some problems)
        # body_sort(ast_tree)


        return ast.unparse(ast_tree)


def difference(pair: tuple[str, str]) -> float:
    lev_d = levenshtein_distance(pair[0], pair[1])
    avg_len = 0.5 * (len(pair[0]) + len(pair[1]))
    diff = lev_d / avg_len
    return diff


def levenshtein_distance(t1: str, t2: str) -> int:
    F = np.zeros((len(t1) + 1, len(t2) + 1))
    F[0] = np.arange(len(t2) + 1)
    F[:, 0] = np.arange(len(t1) + 1)

    for i in range(1, len(t1) + 1):
        for j in range(1, len(t2) + 1):
            if t1[i - 1] == t2[j - 1]:
                F[i][j] = F[i - 1][j - 1]
            else:
                F[i][j] = 1 + min(F[i - 1][j], F[i][j - 1], F[i - 1][j - 1])
    return int(F[len(t1)][len(t2)])


def main(f1: str, f2: str) -> float:
    prep_files = prep_text_code(f1), prep_text_code(f2)
    result = difference(prep_files)

    return result
