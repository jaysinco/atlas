import 'package:flutter/material.dart';
import 'dart:math' as math;

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
    return Column(
      children: [
        Flex(
          direction: Axis.horizontal,
          children: [
            Expanded(
              flex: 1,
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
        Spacer(
          flex: 1,
        ),
        ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
            },
            child: Icon(Icons.arrow_back))
      ],
    );
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

void main() {
  runApp(
    MaterialApp(
      initialRoute: "/",
      routes: {
        "/": (context) => Screen(),
        "test_input": (context) => TestInputRoute(),
        "test_layout": (context) => TestLayoutRoute(),
      },
    ),
  );
}
