import 'package:flutter/material.dart';
import 'dart:math' as math;
import 'package:english_words/english_words.dart';

class StatelessContainer extends StatelessWidget {
  final double width;
  final double height;
  final Color color;

  StatelessContainer({super.key, required this.height, required this.width})
      : color = Color((math.Random().nextDouble() * 0xFFFFFF).toInt())
            .withValues(alpha: 1.0);

  @override
  Widget build(BuildContext context) {
    return Container(width: width, height: height, color: color);
  }
}

class StatefulContainer extends StatefulWidget {
  final double width;
  final double height;

  const StatefulContainer(
      {super.key, required this.height, required this.width});

  @override
  State<StatefulContainer> createState() => _StatefulContainerState();
}

class _StatefulContainerState extends State<StatefulContainer> {
  final Color color;

  _StatefulContainerState()
      : color = Color((math.Random().nextDouble() * 0xFFFFFF).toInt())
            .withValues(alpha: 1.0);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: widget.width,
      height: widget.height,
      color: color,
    );
  }
}

class Screen extends StatefulWidget {
  const Screen({super.key});

  @override
  State<Screen> createState() {
    return _ScreenState();
  }
}

class _ScreenState extends State<Screen> {
  List<Widget> widgets = [
    StatefulContainer(key: UniqueKey(), width: 100, height: 100),
    StatefulContainer(key: UniqueKey(), width: 150, height: 150),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [...widgets],
            ),
            Container(
              height: 20,
            ),
            Padding(
              padding: EdgeInsets.symmetric(vertical: 5),
              child: ElevatedButton(
                onPressed: () {
                  Navigator.pushNamed(context, "test_input");
                },
                child: Text("test_input"),
              ),
            ),
            Padding(
              padding: EdgeInsets.symmetric(vertical: 5),
              child: ElevatedButton(
                onPressed: () {
                  Navigator.pushNamed(context, "test_layout");
                },
                child: Text("test_layout"),
              ),
            ),
            Padding(
              padding: EdgeInsets.symmetric(vertical: 5),
              child: ElevatedButton(
                onPressed: () {
                  Navigator.pushNamed(context, "test_scroll");
                },
                child: Text("test_scroll"),
              ),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: onPressed,
        child: Icon(Icons.undo),
      ),
    );
  }

  onPressed() {
    setState(() {
      widgets.insert(0, widgets.removeAt(1));
    });
  }
}

class TestLayoutRoute extends StatelessWidget {
  const TestLayoutRoute({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Flex(
            direction: Axis.horizontal,
            children: [
              Expanded(
                flex: 4,
                child: Container(
                  height: 30,
                  color: Colors.red,
                ),
              ),
              Expanded(
                flex: 2,
                child: Container(
                  height: 30,
                  color: Colors.green,
                ),
              ),
            ],
          ),
          Padding(
            padding: EdgeInsets.only(top: 10),
            child: SizedBox(
              height: 100,
              child: Flex(
                direction: Axis.vertical,
                children: [
                  Expanded(
                    flex: 2,
                    child: Container(
                      color: Colors.red,
                    ),
                  ),
                  Spacer(
                    flex: 1,
                  ),
                  Expanded(
                    flex: 1,
                    child: Container(
                      color: Colors.green,
                    ),
                  ),
                ],
              ),
            ),
          ),
          Expanded(
            flex: 2,
            child: InfiniteListView(),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
            },
            child: Icon(Icons.arrow_back),
          ),
        ],
      ),
    );
  }
}

class InfiniteListView extends StatefulWidget {
  const InfiniteListView({super.key});

  @override
  State<InfiniteListView> createState() => _InfiniteListViewState();
}

class _InfiniteListViewState extends State<InfiniteListView> {
  static const loadingTag = "##loading##"; //表尾标记

  final _words = <String>[loadingTag];
  final ScrollController _controller = ScrollController();

  @override
  void initState() {
    super.initState();
    _retrieveData();
  }

  @override
  Widget build(BuildContext context) {
    return Scrollbar(
      controller: _controller,
      child: ListView.separated(
        controller: _controller,
        itemCount: _words.length,
        itemBuilder: (context, index) {
          if (_words[index] == loadingTag) {
            if (_words.length - 1 < 100) {
              _retrieveData();
              return Container(
                padding: const EdgeInsets.all(16.0),
                alignment: Alignment.center,
                child: SizedBox(
                  width: 24.0,
                  height: 24.0,
                  child: CircularProgressIndicator(strokeWidth: 2.0),
                ),
              );
            } else {
              return Container(
                alignment: Alignment.center,
                padding: EdgeInsets.all(16.0),
                child: Text(
                  "No More",
                  style: TextStyle(color: Colors.grey),
                ),
              );
            }
          }
          return ListTile(title: Text(_words[index]));
        },
        separatorBuilder: (context, index) => Divider(height: .0),
      ),
    );
  }

  void _retrieveData() {
    Future.delayed(Duration(seconds: 2)).then((e) {
      setState(() {
        _words.insertAll(
          _words.length - 1,
          generateWordPairs().take(20).map((e) => e.asPascalCase).toList(),
        );
      });
    });
  }
}

class TestInputRoute extends StatelessWidget {
  const TestInputRoute({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: EdgeInsets.symmetric(horizontal: 20),
            child: Column(
              children: <Widget>[
                TextField(
                  autofocus: false,
                  decoration: InputDecoration(
                      labelText: "用户名",
                      hintText: "用户名或邮箱",
                      prefixIcon: Icon(Icons.person)),
                ),
                TextField(
                  decoration: InputDecoration(
                      labelText: "密码",
                      hintText: "您的登录密码",
                      prefixIcon: Icon(Icons.lock_clock)),
                  obscureText: true,
                ),
              ],
            ),
          ),
          Text("this is new page"),
          Container(
            height: 10,
          ),
          ElevatedButton.icon(
            onPressed: () {
              Navigator.pop(context);
            },
            label: Text("Back"),
            icon: Icon(
              Icons.arrow_back,
              color: Colors.red,
            ),
          )
        ],
      ),
    );
  }
}

class TestScrollRoute extends StatefulWidget {
  const TestScrollRoute({super.key});

  @override
  State<TestScrollRoute> createState() {
    return _TestScrollRouteState();
  }
}

class _TestScrollRouteState extends State<TestScrollRoute> {
  final ScrollController _controller = ScrollController(keepScrollOffset: true);
  bool showToTopBtn = false;

  @override
  void initState() {
    super.initState();
    _controller.addListener(() {
      if (_controller.offset < 1000 && showToTopBtn) {
        setState(() {
          showToTopBtn = false;
        });
      } else if (_controller.offset >= 1000 && showToTopBtn == false) {
        setState(() {
          showToTopBtn = true;
        });
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("滚动控制")),
      body: Scrollbar(
        controller: _controller,
        child: ListView.builder(
            key: PageStorageKey("test_scroll"),
            itemCount: 100,
            itemExtent: 50.0,
            controller: _controller,
            itemBuilder: (context, index) {
              return ListTile(
                title: Text("$index"),
              );
            }),
      ),
      floatingActionButton: !showToTopBtn
          ? null
          : FloatingActionButton(
              child: Icon(Icons.arrow_upward),
              onPressed: () {
                _controller.animateTo(
                  .0,
                  duration: Duration(milliseconds: 1000),
                  curve: Curves.ease,
                );
              },
            ),
    );
  }
}

void main() {
  runApp(
    MaterialApp(
      initialRoute: "/",
      routes: {
        "/": (context) => Screen(),
        "test_input": (context) => TestInputRoute(),
        "test_layout": (context) => TestLayoutRoute(),
        "test_scroll": (context) => TestScrollRoute(),
      },
    ),
  );
}
