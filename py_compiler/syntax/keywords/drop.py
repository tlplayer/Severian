class Drop:
    def matches(self, node):
        return node.tokens[0].text == "drop"

    def lower(self, node, cfg, env, local_names):
        if len(node.tokens) != 2 or node.tokens[1].text not in env:
            raise ValueError("drop requires a visible binding")
        cfg.flow.drop(env[node.tokens[1].text])
