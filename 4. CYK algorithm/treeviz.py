import nltk
import pydotplus as pdp

import data_structures

def is_tree(t):
    return (isinstance(t, nltk.tree.Tree)
            or isinstance(t, data_structures.ProbabilisticTree))

def escape_label(l):
    needs_quotes = (',' in l or '|' in l)
    return '"%s"' % l if needs_quotes else l

def _tree_to_graph(t, G, ids, token_ids, parent=None):
    ids.append(ids[-1] + 1 if ids else 0)
    this_id = ids[-1]
    if parent is not None:
        G.add_edge(pdp.Edge(parent, this_id))
    if is_tree(t):
        label = escape_label(str(t.label()))
        tooltip = '"%s"' % t.pformat(margin=30)
        G.add_node(pdp.Node(this_id, label=label, tooltip=tooltip))
        for st in t[:]:
            _tree_to_graph(st, G, ids, token_ids, parent=this_id)
    else:
        label = escape_label(str(t))
        G.add_node(pdp.Node(this_id, label=label, tooltip=label, shape='box'))
        token_ids.append(this_id)


def tree_to_graph(t, G=None):
    G = G or pdp.Dot()
    ids = []
    token_ids = []
    _tree_to_graph(t, G, ids, token_ids, parent=None)

    # Make token subgraph to constrain layout
    sg = pdp.Subgraph("tokens", rank="same")
    for token_id in token_ids:
        for node in G.get_node(str(token_id)):
            sg.add_node(node)
    G.add_subgraph(sg)
    # Add token-token edges to preserve sequence
    # Token ids should be sequential, due to left-right traversal.
    for i in range(1, len(token_ids)):
        G.add_edge(pdp.Edge(token_ids[i-1], token_ids[i],
                            arrowsize=0, penwidth=1, color="#CCCCCC"))

    return G

def embed_png_in_html(raw_data):
    import base64
    encoded_data = base64.b64encode(raw_data)
    return "<img src=\"data:image/png;base64," + encoded_data + "\">"

def render_graph(G, format='png', title=None, **graph_attr):
    from IPython.display import display, Image, SVG, HTML
    import base64
    for (k, v) in graph_attr.iteritems():
        G.set(k, v)
    ret = []
    if title:
        ret.append("<h4>" + title + "</h4>")
    if format == 'svg':
        ret.append(G.create_svg(prog='dot'))
    elif format == 'png':
        png_data = G.create_png(prog='dot')
        ret.append(embed_png_in_html(png_data))
    else:
        raise ValueError("Invalid render format " + format)
    return "\n".join(ret)

def make_title(t):
    s = " ".join(t.leaves())
    if hasattr(t, 'logprob'):
        return s + "  (score = %.03f)" % t.logprob()
    return s

def render_tree(t, **kw):
    return render_graph(tree_to_graph(t), **kw)
